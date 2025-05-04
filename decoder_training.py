import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse


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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial projection from 768 to 512 channels
        self.initial_conv = nn.Conv2d(768, 512, 1)

        # Upsampling blocks
        self.decoder = nn.Sequential(
            # 16x16 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1), # Refinement/Attention-like block
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 112x112 -> 224x224
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Final convolution to get 3 channels (RGB)
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensure output is in [0,1] range for images
        )

    def forward(self, x):
        # x shape: (B, 768, 16, 16)
        x = self.initial_conv(x)
        x = self.decoder(x)
        # x shape: (B, 3, 224, 224)
        return x


# --- Dataset ---

class LatentImageDataset(Dataset):
    def __init__(self, latents: torch.Tensor, images: list[Image.Image], target_size=(224, 224)):
        self.latents = latents
        self.images = torch.tensor(np.array([np.array(image) for image in images])).permute(0, 3, 1, 2).float() / 255.0
        self.target_size = target_size # DINOv2 default input size
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

    # List images
    print(f"Scanning for images in {args.image_dir}...")
    image_paths = list(Path(args.image_dir).glob('*.jpg'))
    images = [Image.open(path).convert('RGB') for path in image_paths]
    print(f"Loaded {len(image_paths)} images.")
    if not image_paths:
        print("No images found. Exiting.")
        return

    if args.cache_latents and Path(args.cache_dir).exists():
        dinov2_latents = torch.load(Path(args.cache_dir) / 'dinov2_latents.pt')
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
            print(f"Caching DINOv2 latents to {args.cache_dir}...")
            Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
            torch.save(dinov2_latents, Path(args.cache_dir) / 'dinov2_latents.pt')

    # Create dataset and dataloader
    print("Creating dataset and dataloader...")
    # Ensure target size matches DINOv2 standard input
    dataset = LatentImageDataset(dinov2_latents, images, target_size=(224, 224))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == torch.device("cuda") else False
    )

    # Initialize Decoder model
    decoder = Decoder().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)

    # Training loop
    print("Starting training...")
    decoder.train()
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
                loss = criterion(output_images, target_images_batch)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Update progress bar
                if i % 10 == 9: # Log every 10 batches
                    progress_bar.set_postfix({'loss': f'{running_loss / 10:.4f}'})
                    running_loss = 0.0

        print('Finished Training')
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model...")
    except Exception as e:
        print(f"Error: {e}")
        print("Training interrupted by error. Saving model...")
    finally:
        # Save the trained decoder model
        if args.output_model_path:
            print(f"Saving model to {args.output_model_path}")
            torch.save(decoder.state_dict(), args.output_model_path)
            print("Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decoder model on DINOv2 latents.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing training images.")
    parser.add_argument("--dinov2_model", type=str, default='facebook/dinov2-base', help="DINOv2 model name from Hugging Face.")
    parser.add_argument("--cache-latents", action=argparse.BooleanOptionalAction, default=True, help="Cache DINOv2 latents (default: True). Use --no-cache-latents to disable.")
    parser.add_argument("--cache_dir", type=str, default='dinov2_latents', help="Directory to cache DINOv2 latents.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and latent extraction.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for the DataLoader.")
    parser.add_argument("--output_model_path", type=str, default="decoder_model.pth", help="Path to save the trained decoder model.")

    args = parser.parse_args()
    train_decoder(args)
