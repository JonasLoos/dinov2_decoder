import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel


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
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            # Initial projection
            nn.Conv2d(768, 512, kernel_size=1),

            # Block 1: 16x16 -> 28x28
            Upsample2d(512, 256, padding=0),
            nn.GroupNorm(256, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(256, 256),

            # Block 2: 28x28 -> 56x56
            Upsample2d(256, 128),
            nn.GroupNorm(128, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(128, 128),

            # Block 3: 56x56 -> 112x112
            Upsample2d(128, 64),
            nn.GroupNorm(64, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(64, 64),

            # Block 4: 112x112 -> 224x224
            Upsample2d(64, 32),
            nn.GroupNorm(32, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(32, 32),

            # Final convolution to get 3 channels (RGB)
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()  # Ensure output is in [0,1] range for images
        )

    def forward(self, x):
        # x shape: (B, 768, 16, 16)
        x = self.decoder(x)
        # x shape: (B, 3, 224, 224)
        return x


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

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=args
    )

    # List images
    print(f"Scanning for images in {args.image_dir}...")
    image_paths = list(Path(args.image_dir).glob('*.jpg'))
    images = [Image.open(path).convert('RGB') for path in image_paths]
    print(f"Loaded {len(image_paths)} images.")
    if not image_paths:
        print("No images found. Exiting.")
        return

    # Load cached latents if available
    if args.cache_latents and Path(args.cache_dir).exists():
        dinov2_latents = torch.load(Path(args.cache_dir) / 'dinov2_latents.pt', weights_only=True)
        if len(dinov2_latents) != len(image_paths):
            print(f"Error: cached latents don't match image data")
            return
        print(f"Loaded DINOv2 latents from {args.cache_dir}.")
    else:
        # Load DINOv2 model and processor
        print("Loading DINOv2 model and processor...")
        processor = AutoImageProcessor.from_pretrained(args.dinov2_model)
        dinov2_model = AutoModel.from_pretrained(args.dinov2_model)
        dinov2_latents = get_dinov2_latents(image_paths, processor, dinov2_model, device, args.batch_size)
        if args.cache_latents:
            print(f"Caching DINOv2 latents to folder `{args.cache_dir}` ...")
            Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
            torch.save(dinov2_latents, Path(args.cache_dir) / 'dinov2_latents.pt')

    # Create dataset and dataloader
    print("Creating dataset and dataloader...")
    dataset = LatentImageDataset(dinov2_latents, images)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == torch.device("cuda") else False
    )

    # Create a non-shuffled dataloader for deterministic visualization batch
    vis_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # Ensure deterministic batch selection
        num_workers=args.num_workers,
        pin_memory=True if device == torch.device("cuda") else False
    )

    # Get a fixed batch for visualization
    fixed_latents_batch, fixed_target_images_batch = next(iter(vis_dataloader))
    fixed_latents_batch = fixed_latents_batch.to(device)
    fixed_target_images_batch = fixed_target_images_batch.to(device)

    # Initialize Decoder model
    decoder = torch.compile(Decoder().to(device))

    # Watch model parameters and gradients with wandb
    wandb.watch(decoder, log='all', log_freq=100) # Log gradients every 100 batches

    # Loss and optimizer
    optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # EMA Model
    ema_model = AveragedModel(decoder, device=device, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: args.ema_decay * averaged_model_parameter + (1 - args.ema_decay) * model_parameter)

    # Schedulers
    warmup_iters = args.warmup_epochs * len(dataloader)
    total_iters = args.epochs * len(dataloader)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.001, total_iters=warmup_iters)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=args.learning_rate * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_iters])

    # Training loop
    print("Starting training...")
    decoder.train()
    ema_model.train()
    try:
        for epoch in range(args.epochs):
            running_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

            for i, (latents_batch, target_images_batch) in enumerate(progress_bar):
                latents_batch = latents_batch.to(device)
                target_images_batch = target_images_batch.to(device)

                optimizer.zero_grad()

                # Forward pass
                output_images = decoder(latents_batch)

                # Calculate loss
                l1_loss = F.l1_loss(output_images, target_images_batch)
                l2_loss = F.mse_loss(output_images, target_images_batch)
                loss = l1_loss + l2_loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                ema_model.update_parameters(model=decoder) # Update EMA weights
                scheduler.step() # Update learning rate

                # Log loss and LR to wandb
                psnr = -10 * torch.log10(l2_loss)
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "l1_loss": l1_loss.item(),
                    "l2_loss": l2_loss.item(),
                    "psnr": psnr.item(),
                    "learning_rate": current_lr,
                    "epoch": epoch,
                })

                running_loss += l1_loss.item()

                # Update progress bar
                if i % 10 == 9: # Log every 10 batches
                    avg_loss = running_loss / 10
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    running_loss = 0.0

            # Log sample reconstructions at the end of each epoch using EMA model
            ema_model.eval()
            with torch.no_grad():
                # Generate reconstructions
                output_images_ema = ema_model(fixed_latents_batch)
            ema_model.train()

            # Prepare images for logging (log first 4 images from the fixed batch)
            log_images = []
            num_samples_to_log = min(4, args.batch_size)
            for j in range(num_samples_to_log):
                original = fixed_target_images_batch[j].cpu().permute(1, 2, 0).numpy()
                reconstructed = output_images_ema[j].cpu().permute(1, 2, 0).numpy()
                log_images.append(wandb.Image(original, caption=f"Epoch {epoch+1} - Original {j}"))
                log_images.append(wandb.Image(reconstructed, caption=f"Epoch {epoch+1} - EMA Reconstructed {j}"))

            wandb.log({
                "epoch": epoch + 1,
                "sample_reconstructions": log_images,
            }, step=(epoch+1) * len(dataloader))

        print('Finished Training')
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model...")
    except Exception as e:
        print(f"Error: {e}")
        print("Training interrupted by error. Saving model...")
    finally:
        # Save the trained EMA decoder model
        if args.output_model_path:
            print(f"Saving EMA model to {args.output_model_path}")
            # The AveragedModel wraps the original model, access the module for the state dict
            torch.save(ema_model.module.state_dict(), args.output_model_path)
            print("Model saved.")
            model_artifact = wandb.Artifact(args.output_model_path, type="model")
            model_artifact.add_file(args.output_model_path)
            wandb.log_artifact(model_artifact)

        # Finish wandb run
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decoder model on DINOv2 latents.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing training images.")
    parser.add_argument("--dinov2_model", type=str, default='facebook/dinov2-base', help="DINOv2 model name from Hugging Face.")
    parser.add_argument("--cache-latents", action=argparse.BooleanOptionalAction, default=True, help="Cache DINOv2 latents (default: True). Use --no-cache-latents to disable.")
    parser.add_argument("--cache_dir", type=str, default='dinov2_latents', help="Directory to cache DINOv2 latents.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and latent extraction.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for the DataLoader.")
    parser.add_argument("--output_model_path", type=str, default="decoder_model.pth", help="Path to save the trained decoder model.")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay for the Adam optimizer.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of epochs for linear learning rate warmup.")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="Decay factor for Exponential Moving Average of model weights.")

    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="dinov2_decoder", help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team name).")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name.")

    args = parser.parse_args()
    train_decoder(args)
