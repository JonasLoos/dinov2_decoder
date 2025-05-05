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
import random
import math # Added for Transformer Decoder


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


# --- Helper Modules for Transformer Decoder ---

class PositionalEncoding2D(nn.Module):
    """Adds 2D Sinusoidal Positional Encoding to the input tensor."""
    def __init__(self, d_model, height, width):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with "
                             f"even dimension (got d_model={d_model})") # Corrected error message
        pe = torch.zeros(d_model, height, width)
        # Each dimension uses half of d_model
        d_model_half = d_model // 2
        # Correct calculation for div_term based on standard implementations
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / (d_model_half - 2))) # Adjusted divisor
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        # Apply sin/cos to even/odd indices respectively
        pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        # Add batch dimension and register as buffer
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False) # Add persistent=False if not part of state_dict

    def forward(self, x):
        # x shape: [B, C, H, W]
        # Add positional encoding, slicing pe if x has different H/W than initialized
        # Ensure pe is on the same device as x
        return x + self.pe[:, :, :x.size(2), :x.size(3)].to(x.device)

class TransformerDecoderBlock(nn.Module):
    """Standard Transformer Encoder Block (Self-Attention + MLP)."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu # Using GELU which is common in Transformers

    def forward(self, src):
        # src shape: [B, SeqLen, Dim] (batch_first=True)
        # Self-attention part with pre-normalization (common practice)
        src_norm = self.norm1(src)
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm) # Q=K=V for self-attention
        src = src + self.dropout1(attn_output) # Residual connection
        # Feedforward part with pre-normalization
        src_norm = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_output) # Residual connection
        return src

class UpsampleBlock(nn.Module):
    """Upsamples input, applies Conv, Norm (optional), and ReLU."""
    def __init__(self, in_channels, out_channels, scale_factor=2, use_norm=True):
        super().__init__()
        # Use bilinear upsampling followed by convolution to avoid checkerboard artifacts
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Use GroupNorm as an alternative to BatchNorm, often works well with Transformers
        # Using 1 group makes it LayerNorm like across channels, using C groups makes it InstanceNorm like
        num_groups = min(32, out_channels // 4) if out_channels >= 32 else out_channels # Heuristic for num_groups
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels) if use_norm and out_channels > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True) # ReLU is fine, could also use GELU

    def forward(self, x):
        x = self.upsample(x)
        x = x.contiguous() # Ensure contiguous memory layout after upsampling
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


# --- Transformer-based Decoder ---

class TransformerPixelDecoder(nn.Module):
    """
    Decodes DINOv2 latents (patch embeddings) into an image using Transformer blocks
    followed by a CNN upsampling path.

    Args:
        input_dim (int): Dimension of input DINOv2 features (e.g., 768).
        hidden_dim (int): Internal dimension used by the Transformer and initial CNN layers.
        nhead (int): Number of attention heads in Transformer blocks.
        num_transformer_layers (int): Number of Transformer blocks.
        dim_feedforward (int): Dimension of the feedforward network in Transformer blocks.
        input_res (tuple): Spatial resolution of the input features (H, W), e.g., (16, 16).
        output_res (tuple): Target spatial resolution of the output image (H, W), e.g., (224, 224).
        use_norm_in_upsample (bool): Whether to use normalization (GroupNorm) in the upsampling blocks.
    """
    def __init__(self, input_dim=768, hidden_dim=384, nhead=6, num_transformer_layers=4,
                 dim_feedforward=1536, input_res=(16, 16), output_res=(224, 224), use_norm_in_upsample=True):
        super().__init__()
        self.input_res = input_res
        self.output_res = output_res
        self.hidden_dim = hidden_dim
        if hidden_dim % nhead != 0:
             raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})")

        # 1. Initial projection to hidden dimension (maintains spatial resolution)
        self.proj_in = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        # 2. Positional encoding for spatial information
        self.pos_encoder = PositionalEncoding2D(hidden_dim, input_res[0], input_res[1])

        # 3. Transformer blocks to process patch embeddings
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderBlock(hidden_dim, nhead, dim_feedforward)
            for _ in range(num_transformer_layers)
        ])

        # 4. CNN Upsampling Path
        # Calculate necessary upsampling stages. Target scale factor = output_res / input_res
        target_scale_factor = output_res[0] / input_res[0]
        num_upsample_layers = math.ceil(math.log2(target_scale_factor)) # Number of 2x upsampling stages needed

        upsample_modules = []
        current_dim = hidden_dim
        # Progressively reduce channel dimension by halving, ensuring divisibility by 8 and minimum of 32
        # ch_exp = np.linspace(np.log2(hidden_dim), np.log2(max(32, hidden_dim // (2**num_upsample_layers))), num_upsample_layers + 1)
        # channel_dims = [int(2**exp) for exp in ch_exp]
        channel_dims = [hidden_dim]
        temp_current_dim = hidden_dim
        for _ in range(num_upsample_layers):
            next_dim = max(32, temp_current_dim // 2)
            # Ensure divisibility by 8 for compatibility with GroupNorm heuristic, minimum 32
            next_dim = max(32, (next_dim // 8) * 8 if next_dim >= 32 else next_dim)
            channel_dims.append(next_dim)
            temp_current_dim = next_dim

        for i in range(num_upsample_layers):
            in_ch = channel_dims[i]
            out_ch = channel_dims[i+1]
            upsample_modules.append(UpsampleBlock(in_ch, out_ch, scale_factor=2, use_norm=use_norm_in_upsample))
            current_dim = out_ch

        self.upsampling_cnn = nn.Sequential(*upsample_modules)

        # 5. Final layer to match output resolution and channels
        # Current resolution after num_upsample_layers of 2x scaling
        current_res = input_res[0] * (2**num_upsample_layers)
        # Add final interpolation if needed (e.g., if target 224 != 16 * 2^4 = 256)
        self.needs_final_interp = current_res != output_res[0]
        if self.needs_final_interp:
            self.final_upsample = nn.Upsample(size=output_res, mode='bilinear', align_corners=False)
            print(f"Decoder: Adding final interpolation from {current_res}x{current_res} to {output_res[0]}x{output_res[1]}")
        else:
            self.final_upsample = nn.Identity()

        # Final 1x1 conv to get 3 channels (RGB)
        self.final_conv = nn.Conv2d(current_dim, 3, kernel_size=1)
        self.final_activation = nn.Sigmoid() # Output pixel values in [0, 1]


    def forward(self, x):
        # Input x: [B, input_dim, H, W], e.g., [B, 768, 16, 16] (DINOv2 patch embeddings)
        B, C, H, W = x.shape
        if H != self.input_res[0] or W != self.input_res[1]:
             # Allow for slight variation if needed, or raise error
             # For now, we'll resize the input if it doesn't match exactly.
             # This might happen if a different DINO model is used.
             print(f"Warning: Input spatial resolution mismatch. Expected {self.input_res}, got {(H, W)}. Resizing input.")
             x = F.interpolate(x, size=self.input_res, mode='bilinear', align_corners=False)
             B, C, H, W = x.shape # Update H, W after resize

        # 1. Projection
        x = self.proj_in(x) # [B, hidden_dim, H, W]

        # 2. Add positional encoding
        x_pos = self.pos_encoder(x) # [B, hidden_dim, H, W]

        # 3. Flatten and run through Transformer blocks
        # Flatten: [B, hidden_dim, H*W] -> Permute: [B, H*W, hidden_dim]
        x_flat = x_pos.flatten(2).permute(0, 2, 1)
        for layer in self.transformer_layers:
            x_flat = layer(x_flat) # [B, H*W, hidden_dim]

        # 4. Reshape back to image grid
        # Permute: [B, hidden_dim, H*W] -> Reshape: [B, hidden_dim, H, W]
        x_tf_out = x_flat.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W).contiguous()

        # 5. Upsample using CNN
        x_up = self.upsampling_cnn(x_tf_out) # Output size depends on num_upsample_layers

        # 6. Final layers
        x_final = self.final_upsample(x_up) # Interpolate to target size if needed
        x_final = self.final_conv(x_final) # [B, 3, output_res[0], output_res[1]]
        output = self.final_activation(x_final) # [B, 3, output_res[0], output_res[1]]

        return output

# --- Original Decoder (commented out for reference) ---
# class ConvWithSkip(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.nn = nn.Sequential(
#             nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(num_features, num_features, kernel_size=1),
#             nn.GELU(),
#         )
#
#     def forward(self, x):
#         return self.nn(x) + x
#
# class Upsample2d(nn.Module):
#     def __init__(self, in_channels, out_channels, padding=1):
#         super().__init__()
#         # Correct implementation: Conv first, then PixelShuffle (or TransposedConv)
#         # This version used a Conv to increase channels then reshaped - like PixelShuffle
#         # Let's keep it similar for now but note TransposedConv is another option
#         self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=padding)
#         self.pixel_shuffle = nn.PixelShuffle(2) # Upscales H, W by factor of 2
#
#     def forward(self, x):
#         x = self.conv(x) # [B, C_out*4, H, W]
#         x = self.pixel_shuffle(x) # [B, C_out, H*2, W*2]
#         x = F.gelu(x) # Apply activation *after* shuffle
#         return x
#
# Decoder = lambda: nn.Sequential(
#     # Initial projection
#     nn.Conv2d(768, 256, kernel_size=1),
#
#     # Block 1: 16x16 -> 32x32 (Mistake in original comment: 16->28 not possible with std 2x upsample)
#     # Assuming padding=1 was intended for Upsample2d to make it 32x32
#     Upsample2d(256, 128, padding=1), # Output: 128 x 32 x 32
#     ConvWithSkip(128),
#
#     # Block 2: 32x32 -> 64x64
#     Upsample2d(128, 64, padding=1), # Output: 64 x 64 x 64
#     ConvWithSkip(64),
#
#     # Block 3: 64x64 -> 128x128
#     Upsample2d(64, 32, padding=1), # Output: 32 x 128 x 128
#     ConvWithSkip(32),
#
#     # Block 4: 128x128 -> 256x256
#     Upsample2d(32, 16, padding=1), # Output: 16 x 256 x 256
#     ConvWithSkip(16),
#
#     # Final convolution needs adjustment if output is 256x256 and target is 224x224
#     # Option 1: Add interpolation/cropping
#     # Option 2: Adjust last Upsample2d or add a ConvTranspose2d with specific stride/padding
#     # Adding an adaptive pool for simplicity to get to 224x224 before final conv
#     nn.AdaptiveAvgPool2d((224, 224)),
#
#     nn.Conv2d(16, 3, kernel_size=1),
#     nn.Sigmoid()  # Ensure output is in [0,1] range for images
# )

# --- Instantiate the New Decoder ---
# You can adjust the hyperparameters here
Decoder = lambda: TransformerPixelDecoder(
    input_dim=768,          # DINOv2 feature dimension
    hidden_dim=384,         # Internal dimension (multiple of nhead)
    nhead=6,                # Number of attention heads
    num_transformer_layers=4,# Number of transformer blocks
    dim_feedforward=1536,   # Feedforward dimension in transformer
    input_res=(16, 16),     # DINOv2 feature map size (check based on model)
    output_res=(224, 224),  # Target image size
    use_norm_in_upsample=True
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
    all_image_paths = sorted(list(Path(args.image_dir).glob('*.jpg'))) # Sort for consistent splits
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

    # visualization data
    fixed_latents_train, fixed_images_train = train_dataset[:args.batch_size]
    fixed_latents_train = fixed_latents_train.to(device)
    fixed_latents_test, fixed_images_test = test_dataset[:args.batch_size]
    fixed_latents_test = fixed_latents_test.to(device)

    # Initialize Decoder model
    decoder = Decoder().to(device)
    # decoder = torch.compile(decoder)

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
                loss = l1_loss + l2_loss

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
            with torch.no_grad():
                for latents_batch, target_images_batch in tqdm(test_dataloader, desc=f"Epoch {epoch + 1} Test Eval"):
                    latents_batch = latents_batch.to(device)
                    target_images_batch = target_images_batch.to(device)

                    # Forward pass with EMA model
                    output_images_ema = eval_model(latents_batch)

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

            wandb.log({
                "test/l1_loss": avg_test_l1,
                "test/l2_loss": avg_test_l2,
                "test/psnr": avg_test_psnr,
                "epoch": epoch + 1, # Log test metrics against epoch number
                "step": global_step # Log against global step as well
            })
            print(f"Epoch {epoch + 1} Test Results: L1={avg_test_l1:.4f}, L2={avg_test_l2:.4f}, PSNR={avg_test_psnr:.4f}")


            # --- Log sample reconstructions (using EMA model) ---
            with torch.no_grad():
                # Generate reconstructions using the fixed test batch
                output_images_train = eval_model(fixed_latents_train)
                output_images_test = eval_model(fixed_latents_test)

            # Prepare images for logging (log first 4 images from the fixed test batch)
            log_images_train = []
            log_images_test = []
            num_samples_to_log = min(4, args.batch_size)
            for j in range(num_samples_to_log):
                original_train = fixed_images_train[j].cpu().permute(1, 2, 0).numpy()
                original_test = fixed_images_test[j].cpu().permute(1, 2, 0).numpy()
                reconstructed_train = output_images_train[j].cpu().permute(1, 2, 0).clamp(0, 1).numpy() # Clamp output just in case
                reconstructed_test = output_images_test[j].cpu().permute(1, 2, 0).clamp(0, 1).numpy() # Clamp output just in case
                log_images_train.append(wandb.Image(original_train, caption=f"Epoch {epoch+1} - Train Original {j}"))
                log_images_test.append(wandb.Image(original_test, caption=f"Epoch {epoch+1} - Test Original {j}"))
                log_images_train.append(wandb.Image(reconstructed_train, caption=f"Epoch {epoch+1} - Train Reconstructed {j}"))
                log_images_test.append(wandb.Image(reconstructed_test, caption=f"Epoch {epoch+1} - Test Reconstructed {j}"))

            wandb.log({
                "train/sample_reconstructions": log_images_train,
                "test/sample_reconstructions": log_images_test,
                "epoch": epoch + 1,
                "step": global_step
            })

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
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing training images.")
    parser.add_argument("--dinov2_model", type=str, default='facebook/dinov2-base', help="DINOv2 model name from Hugging Face.")
    parser.add_argument("--cache-latents", action=argparse.BooleanOptionalAction, default=True, help="Cache DINOv2 latents (default: True). Use --no-cache-latents to disable.")
    parser.add_argument("--cache_dir", type=str, default='dinov2_latents', help="Directory to cache DINOv2 latents.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and latent extraction.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for the DataLoader.")
    parser.add_argument("--output_model_path", type=str, default="decoder_model.pth", help="Path to save the trained decoder model.")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay for the Adam optimizer.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of epochs for linear learning rate warmup.")
    parser.add_argument("--ema_decay", type=float, default=0, help="Decay factor for Exponential Moving Average of model weights. Set to 0 to disable EMA.")
    parser.add_argument("--test_split_ratio", type=float, default=0.1, help="Fraction of data to use for the test set (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="dinov2_decoder", help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team name).")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name.")

    args = parser.parse_args()
    train_decoder(args)
