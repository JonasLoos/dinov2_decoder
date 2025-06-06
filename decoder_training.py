import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
import random
import matplotlib.pyplot as plt


# --- Reproducibility ---

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # Ensure deterministic behavior for certain CUDA operations
    # Note: This might impact performance.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# --- DINOv2 Feature Extraction ---

def get_dinov2_latents(image_paths, processor, model, device, batch_size=32):
    """Extracts DINOv2 latents for a list of images."""
    model.eval()
    model.to(device)

    # Trace the model for potential performance improvement
    model.config.return_dict = False
    try:
        with torch.no_grad():
            sample_image = Image.open(image_paths[0]).convert('RGB')
            inputs = processor(images=sample_image, return_tensors="pt").to(device)
            traced_model = torch.jit.trace(model, [inputs.pixel_values])
    except Exception as e:
        print(f"Could not trace model, using original model. Error: {e}")
        traced_model = model

    all_latents = []
    print("Extracting DINOv2 latents...")
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting DINOv2 latents"):
        batch_paths = image_paths[i:i+batch_size]
        # Ensure images are RGB
        batch_images = [Image.open(path).convert('RGB') for path in batch_paths]

        with torch.no_grad():
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            # Output shape: [batch_size, num_patches + 1, hidden_size]
            # We discard the CLS token [:, 0, :] and take patch tokens [:, 1:, :]
            # Reshape to [batch_size, 16, 16, hidden_size] (assuming 16x16 patches)
            outputs = traced_model(inputs.pixel_values)
            # Handle potential tuple output from traced model
            latents = outputs[0] if isinstance(outputs, tuple) else outputs
            latents = latents[:, 1:, :].reshape(-1, 16, 16, 768)
        all_latents.append(latents.cpu())

    return torch.cat(all_latents, dim=0).permute(0, 3, 1, 2)


# --- Decoder Model ---

class ConvWithSkip(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_features, num_features, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.nn(x) + x

class Upsample2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels*4, kernel_size=3, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c//4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(b, c//4, h*2, w*2)
        x = F.gelu(x)
        return x


Decoder = lambda: nn.Sequential(
    # Initial projection
    nn.Conv2d(768, 256, kernel_size=1),

    # Block 1: 16x16 -> 28x28
    Upsample2d(256, 128, padding=0),
    ConvWithSkip(128),

    # Block 2: 28x28 -> 56x56
    Upsample2d(128, 64),
    ConvWithSkip(64),

    # Block 3: 56x56 -> 112x112
    Upsample2d(64, 32),
    ConvWithSkip(32),

    # Block 4: 112x112 -> 224x224
    Upsample2d(32, 16),
    ConvWithSkip(16),

    # Final convolution to get 3 channels (RGB)
    nn.Conv2d(16, 3, kernel_size=1),
    nn.Sigmoid()  # Ensure output is in [0,1] range for images
)


# --- Dataset ---

class LatentImageDataset(Dataset):
    def __init__(self, latents: torch.Tensor, images: list[Image.Image]):
        self.latents = latents
        self.images = torch.tensor(np.array([np.array(image) for image in images])).permute(0, 3, 1, 2).float() / 255.0
        if len(latents) != len(images):
             raise ValueError("Number of latents and image paths must be the same.")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.images[idx]


# --- Training ---

def train_decoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=args
    )

    # List images
    print(f"Scanning for images in {args.image_dir}...")
    all_image_paths = list(Path(args.image_dir).glob('*.jpg'))
    print(f"Found {len(all_image_paths)} images.")
    if not all_image_paths:
        print("No images found. Exiting.")
        return

    # Load cached latents if available
    cache_file = Path(args.cache_dir) / 'dinov2_latents.pt'
    if args.cache_latents and cache_file.exists():
        dinov2_latents = torch.load(cache_file, weights_only=True)
        if len(dinov2_latents) != len(all_image_paths):
            print(f"Error: cached latents length ({len(dinov2_latents)}) doesn't match image count ({len(all_image_paths)}). Recalculating.")
            dinov2_latents = None # Force recalculation
        else:
            print(f"Loaded DINOv2 latents from {cache_file}.")
    else:
        dinov2_latents = None

    if dinov2_latents is None:
        # Load DINOv2 model and processor
        print("Loading DINOv2 model and processor...")
        processor = AutoImageProcessor.from_pretrained(args.dinov2_model)
        dinov2_model = AutoModel.from_pretrained(args.dinov2_model)
        # Load all images *before* latent extraction
        all_images = [Image.open(path).convert('RGB') for path in all_image_paths]
        dinov2_latents = get_dinov2_latents(all_image_paths, processor, dinov2_model, device, args.batch_size)
        if args.cache_latents:
            print(f"Caching DINOv2 latents to folder `{args.cache_dir}` ...")
            Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
            torch.save(dinov2_latents, cache_file)
    else:
        # Load images only if latents were cached successfully
        print("Loading images...")
        all_images = [Image.open(path).convert('RGB') for path in tqdm(all_image_paths, desc="Loading Images")]


    # --- Train/Test Split ---
    print(f"Splitting data into train/test sets (test ratio: {args.test_split_ratio})...")
    indices = list(range(len(all_image_paths)))
    test_size = int(len(indices) * args.test_split_ratio)

    # test/train split
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(indices), generator=generator)
    test_indices = [indices[i] for i in perm[:test_size]]
    train_indices = [indices[i] for i in perm[test_size:]]

    train_latents = dinov2_latents[train_indices]
    test_latents = dinov2_latents[test_indices]
    train_images = [all_images[i] for i in train_indices]
    test_images = [all_images[i] for i in test_indices]

    print(f"Train set size: {len(train_images)}")
    print(f"Test set size: {len(test_images)}")

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = LatentImageDataset(train_latents, train_images)
    test_dataset = LatentImageDataset(test_latents, test_images)

    # Use a generator for reproducible shuffling in training dataloader
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == torch.device("cuda") else False,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id), # For reproducibility
        generator=g
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=args.num_workers,
        pin_memory=True if device == torch.device("cuda") else False
    )

    # Initialize Decoder model
    decoder = Decoder().to(device)
    decoder = torch.compile(decoder)

    # Watch model parameters and gradients with wandb
    wandb.watch(decoder, log='all', log_freq=100) # Log gradients every 100 batches

    # Loss and optimizer
    optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # EMA Model
    ema_model = None
    if args.ema_decay > 0:
        ema_model = AveragedModel(decoder, device=device, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: args.ema_decay * averaged_model_parameter + (1 - args.ema_decay) * model_parameter)
        print(f"EMA enabled with decay: {args.ema_decay}")

    # Schedulers
    warmup_iters = args.warmup_epochs * len(train_dataloader)
    total_iters = args.epochs * len(train_dataloader)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_iters)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=args.learning_rate * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_iters])

    # Training loop
    print("Starting training...")
    try:
        global_step = 0 # Track total steps for logging
        for epoch in range(args.epochs):
            decoder.train() # Ensure model is in training mode
            if ema_model:
                ema_model.train() # Ensure EMA model is in training mode
            running_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}") # Use train_dataloader

            for i, (latents_batch, target_images_batch) in enumerate(progress_bar):
                latents_batch = latents_batch.to(device)
                target_images_batch = target_images_batch.to(device)

                optimizer.zero_grad()

                # Forward pass (standard model)
                output_images = decoder(latents_batch)

                # Calculate loss
                l1_loss = F.l1_loss(output_images, target_images_batch)
                l2_loss = F.mse_loss(output_images, target_images_batch)
                loss = args.l1_weight * l1_loss + (1 - args.l1_weight) * l2_loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                if ema_model:
                    ema_model.update_parameters(model=decoder) # Update EMA weights
                scheduler.step() # Update learning rate

                # Log training loss and LR to wandb
                psnr = -10 * torch.log10(l2_loss)
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "train/l1_loss": l1_loss.item(),
                    "train/l2_loss": l2_loss.item(),
                    "train/loss": loss.item(),
                    "train/psnr": psnr.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch,
                    "step": global_step
                })
                global_step += 1

                running_loss += loss.item() # Use combined loss for progress bar

                # Update progress bar
                if i % 10 == 9: # Log every 10 batches
                    avg_loss = running_loss / 10
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.6f}'})
                    running_loss = 0.0

            # --- Evaluation on Test Set ---
            eval_model = ema_model if ema_model else decoder # Use EMA if available, otherwise use base model
            eval_model.eval() # Set the chosen model to evaluation mode

            test_l1_loss = 0.0
            test_l2_loss = 0.0
            test_psnr = 0.0
            num_test_batches = 0
            # Initialize FID metric
            fid = FrechetInceptionDistance(feature=64).to(device) # Using 64 features for faster computation, can be adjusted
            with torch.no_grad():
                for latents_batch, target_images_batch in tqdm(test_dataloader, desc=f"Epoch {epoch + 1} Test Eval"):
                    latents_batch = latents_batch.to(device)
                    target_images_batch = target_images_batch.to(device)

                    # Forward pass with EMA model
                    output_images_ema = eval_model(latents_batch)

                    # Prepare images for FID (convert to uint8 and scale to 0-255)
                    # FID expects NCHW format
                    real_images_fid = (target_images_batch.clamp(0, 1) * 255).byte()
                    fake_images_fid = (output_images_ema.clamp(0, 1) * 255).byte()

                    # Update FID
                    fid.update(real_images_fid, real=True)
                    fid.update(fake_images_fid, real=False)

                    # Calculate test loss
                    batch_l1 = F.l1_loss(output_images_ema, target_images_batch)
                    batch_l2 = F.mse_loss(output_images_ema, target_images_batch)
                    batch_psnr = -10 * torch.log10(batch_l2)

                    test_l1_loss += batch_l1.item()
                    test_l2_loss += batch_l2.item()
                    test_psnr += batch_psnr.item()
                    num_test_batches += 1

            avg_test_l1 = test_l1_loss / num_test_batches
            avg_test_l2 = test_l2_loss / num_test_batches
            avg_test_psnr = test_psnr / num_test_batches
            test_fid_score = fid.compute()

            def visualize(dataset):
                idx = [0, 12, 42, 80, 100]  # just some random numbers, that are smaller than the dataset length, but not directly next to each other
                latents, images = dataset[idx]
                # generate outputs
                with torch.no_grad():
                    outputs = eval_model(latents.to(device)).cpu()
                # plot
                fig, axs = plt.subplots(2, len(idx), figsize=(15, 6))
                for j, (image, output) in enumerate(zip(images, outputs)):
                    axs[0, j].imshow(image.permute(1, 2, 0).numpy())
                    axs[1, j].imshow(output.permute(1, 2, 0).numpy())
                    axs[0, j].axis('off')
                    axs[1, j].axis('off')
                plt.tight_layout()
                return fig

            wandb.log({
                "test/l1_loss": avg_test_l1,
                "test/l2_loss": avg_test_l2,
                "test/psnr": avg_test_psnr,
                "test/fid": test_fid_score.item(),
                "train/sample_reconstructions": wandb.Image(visualize(train_dataset), caption=f"Epoch {epoch+1} - Train Reconstructions"),
                "test/sample_reconstructions": wandb.Image(visualize(test_dataset), caption=f"Epoch {epoch+1} - Test Reconstructions"),
                "epoch": epoch + 1,
                "step": global_step,
            })
            plt.close()
            print(f"Epoch {epoch + 1} Test Results: L1={avg_test_l1:.4f}, L2={avg_test_l2:.4f}, PSNR={avg_test_psnr:.4f}, FID={test_fid_score.item():.4f}")

        print('Finished Training')
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model...")
    except Exception as e:
        print(f"Error: {e}")
        print("Training interrupted by error. Saving model...")
    finally:
        # Save the trained EMA decoder model
        if args.output_model_path:
            output_dir = Path(args.output_model_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            if ema_model:
                print(f"Saving EMA model to {args.output_model_path}")
                # Access the underlying model module for the state dict
                torch.save(ema_model.module.state_dict(), args.output_model_path)
                model_artifact_name = f"{Path(args.output_model_path).stem}_ema"
            else:
                print(f"Saving base model to {args.output_model_path}")
                torch.save(decoder.state_dict(), args.output_model_path)
                model_artifact_name = Path(args.output_model_path).stem

            print("Model saved.")
            model_artifact = wandb.Artifact(model_artifact_name, type="model")
            model_artifact.add_file(args.output_model_path)
            wandb.log_artifact(model_artifact)

        # Finish wandb run
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decoder model on DINOv2 latents.")
    parser.add_argument("--image_dir", type=str, default='unsplash_lite_image_dataset/training_images/', help="Directory containing training images.")
    parser.add_argument("--dinov2_model", type=str, default='facebook/dinov2-base', help="DINOv2 model name from Hugging Face.")
    parser.add_argument("--cache-latents", action=argparse.BooleanOptionalAction, default=True, help="Cache DINOv2 latents (default: True). Use --no-cache-latents to disable.")
    parser.add_argument("--cache_dir", type=str, default='dinov2_latents', help="Directory to cache DINOv2 latents.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and latent extraction.")
    parser.add_argument("--epochs", type=int, default=42, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for the DataLoader.")
    parser.add_argument("--output_model_path", type=str, default="decoder_model.pth", help="Path to save the trained decoder model.")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay for the Adam optimizer.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of epochs for linear learning rate warmup.")
    parser.add_argument("--ema_decay", type=float, default=0, help="Decay factor for Exponential Moving Average of model weights. Set to 0 to disable EMA.")
    parser.add_argument("--test_split_ratio", type=float, default=0.1, help="Fraction of data to use for the test set (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--l1_weight", type=float, default=0.5, help="Weight for L1 loss in the total loss calculation (L2 weight will be 1 - l1_weight). Default: 0.5")

    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="dinov2_decoder", help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team name).")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name.")

    args = parser.parse_args()
    train_decoder(args)
