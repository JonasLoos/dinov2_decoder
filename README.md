# DINOv2 Image Decoder Training

This project trains a convolutional decoder model to reconstruct images from their DINOv2 feature representations (latents).


## Prepare Dataset

```bash
python download_unsplash.py
```

This will download a subset of the [Unsplash Lite](https://github.com/unsplash/datasets?tab=readme-ov-file#lite-dataset) dataset and generate `224x224` cropped training images. The images will be saved in `unsplash_lite_image_dataset/training_images/`.


## Train Decoder

```bash
python decoder_training.py --image_dir unsplash_lite_image_dataset/training_images/
```

This script will:

* Load the DINOv2 model (`facebook/dinov2-base` by default).
* Extract DINOv2 latents for the images in the specified directory. Latents will be cached by default in `dinov2_latents/`. Use `--no-cache-latents` to disable caching or `--cache_dir` to change the location.
* Train the decoder model to reconstruct the original images from the latents.
* Save the trained model to `decoder_model.pth` (or specify a different path with `--output_model_path`).


## Requirements

```bash
pip install torch torchvision torchaudio transformers Pillow numpy tqdm requests pandas
```
