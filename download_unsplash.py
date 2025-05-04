import requests
import zipfile
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
from PIL import Image
import random
import math
import logging


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Configuration ---
DATA_DIR = Path("unsplash_lite_image_dataset")
RAW_IMAGES_DIR = DATA_DIR / 'images'
TRAINING_IMAGES_DIR = DATA_DIR / 'training_images'
DATASET_URL = "https://unsplash-datasets.s3.amazonaws.com/lite/latest/unsplash-research-dataset-lite-latest.zip"
DOCUMENTS_TO_LOAD = ['photos']#, 'keywords', 'collections', 'conversions', 'colors']
NUM_IMAGES_TO_DOWNLOAD = 100 # Number of raw images to download
CROP_SIZE = 224
MAX_CROPS_PER_IMAGE = 10


# --- Helper Functions ---

def download_and_extract_zip(url: str, target_dir: Path):
    """Downloads and extracts a zip file."""
    zip_filename = target_dir / Path(url).name
    target_dir.mkdir(exist_ok=True, parents=True)

    if zip_filename.exists():
        logging.info(f"Zip file {zip_filename} already exists. Skipping download.")
    else:
        logging.info(f"Downloading dataset from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_filename, 'wb') as f, tqdm(
                desc=zip_filename.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            logging.info(f"Downloaded {zip_filename}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")
            return False

    logging.info(f"Extracting {zip_filename}...")
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        logging.info(f"Extracted files to {target_dir}")
    except zipfile.BadZipFile:
        logging.error(f"Failed to extract {zip_filename}. It might be corrupted.")
        return False
    return True


def load_tsv_data(data_dir: Path, docs: list) -> dict:
    """Loads TSV files into pandas DataFrames."""
    datasets = {}
    logging.info("Loading TSV data...")
    for doc in docs:
        files = list(data_dir.glob(doc + ".tsv*"))
        if not files:
            logging.warning(f"No TSV files found for document type: {doc}")
            continue

        subsets = []
        for filename in files:
            try:
                df = pd.read_csv(filename, sep='\t', header=0, on_bad_lines='warn')
                subsets.append(df)
            except Exception as e:
                logging.error(f"Failed to read {filename}: {e}")

        if subsets:
            datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)
            logging.info(f"Loaded {doc} data: {datasets[doc].shape[0]} records.")
        else:
             logging.warning(f"Could not load any data for document type: {doc}")
    return datasets


def download_raw_images(image_urls: pd.Series, target_dir: Path):
    """Downloads images from a list of URLs."""
    target_dir.mkdir(exist_ok=True, parents=True)
    fails = []
    logging.info(f"Downloading {len(image_urls)} raw images to {target_dir}...")

    for url in tqdm(image_urls, desc="Downloading images"):
        try:
            # Generate a safe filename from the URL
            name = str(url).split('?')[0].split('/')[-1] + '.jpg' # Basic name from URL path
            if not name: # Handle potential edge cases
                 name = f"image_{random.randint(1000,9999)}.jpg"
            path = target_dir / name

            if path.exists():
                # logging.debug(f"Image {path} already exists. Skipping.")
                continue

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            path.write_bytes(response.content)

        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to download {url}: {e}")
            fails.append(url)
        except Exception as e:
             logging.error(f"An unexpected error occurred for {url}: {e}")
             fails.append(url)

    if fails:
        logging.warning(f"Failed to download {len(fails)} images.")
        # Save the list of failed URLs
        with open(DATA_DIR / "failed_downloads.txt", "w") as f:
            for item in fails:
                f.write(f"{item}\n")
    logging.info("Finished downloading raw images.")


def random_crop(img: Image.Image, size: int) -> Image.Image:
    """Performs a random crop of the specified size."""
    width, height = img.size
    if width < size or height < size:
        # If image is smaller than crop size, resize it up maintaining aspect ratio
        # and then take a center crop. This is one strategy, others are possible.
        ratio = max(size/width, size/height)
        new_w, new_h = int(width * ratio), int(height * ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        width, height = img.size # update dimensions

    left = random.randint(0, width - size)
    top = random.randint(0, height - size)
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))


def generate_training_images(raw_img_dir: Path, training_img_dir: Path, crop_size: int, max_crops: int):
    """Generates downscaled and cropped training images."""
    training_img_dir.mkdir(exist_ok=True, parents=True)
    image_files = list(raw_img_dir.glob('*.jpg'))
    logging.info(f"Generating training images from {len(image_files)} raw images...")

    for filename in tqdm(image_files, desc="Generating training images"):
        try:
            img_raw = Image.open(filename).convert('RGB') # Ensure RGB
            w, h = img_raw.size

            if min(w,h) < crop_size:
                logging.warning(f"Skipping {filename}, smaller than crop size ({w}x{h})")
                continue

            # Calculate maximum possible downscales before image is smaller than crop_size
            max_downscales = math.floor(math.log2(min(w, h) / crop_size))

            for downscale in range(max_downscales + 1): # Include 0 (original size)
                current_w = w // (2**downscale)
                current_h = h // (2**downscale)

                # Ensure downscaled image is still larger than crop size
                if current_w < crop_size or current_h < crop_size:
                    continue

                img_scaled = img_raw if downscale == 0 else img_raw.resize((current_w, current_h), Image.Resampling.LANCZOS)

                # Number of random crops relative to the number of pixels, capped by max_crops
                num_crops = min(max((current_w * current_h) // (crop_size**2), 1), max_crops) # Ensure at least 1 crop

                for i in range(num_crops):
                    try:
                        img_crop = random_crop(img_scaled, crop_size)
                        output_filename = training_img_dir / f'{filename.stem}_{downscale}_{i}.jpg'
                        img_crop.save(output_filename, quality=95)
                    except ValueError as crop_error:
                        logging.error(f"Error cropping {filename} (downscale {downscale}, crop {i}): {crop_error}")
                    except Exception as save_error:
                        logging.error(f"Error saving crop for {filename}: {save_error}")

        except Image.UnidentifiedImageError:
            logging.warning(f"Cannot identify image file {filename}, skipping.")
        except Exception as e:
            logging.error(f"Failed to process {filename}: {e}")

    logging.info("Finished generating training images.")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Download and Extract Dataset
    if not download_and_extract_zip(DATASET_URL, DATA_DIR):
        logging.error("Failed to download or extract dataset. Exiting.")
        exit()

    # 2. Load TSV Data
    datasets = load_tsv_data(DATA_DIR, DOCUMENTS_TO_LOAD)
    if 'photos' not in datasets or datasets['photos'].empty:
         logging.error("Photos data failed to load or is empty. Cannot download images. Exiting.")
         exit()

    # 3. Download Raw Images (Subset)
    if 'photo_image_url' not in datasets['photos'].columns:
        logging.error("'photo_image_url' column not found in photos data. Cannot download images. Exiting.")
        exit()

    image_urls_to_download = datasets['photos']['photo_image_url'].dropna().unique()
    if len(image_urls_to_download) > NUM_IMAGES_TO_DOWNLOAD:
         image_urls_to_download = image_urls_to_download[:NUM_IMAGES_TO_DOWNLOAD]

    download_raw_images(image_urls_to_download, RAW_IMAGES_DIR)

    # 4. Generate Training Images
    generate_training_images(RAW_IMAGES_DIR, TRAINING_IMAGES_DIR, CROP_SIZE, MAX_CROPS_PER_IMAGE)

    logging.info("Dataset preparation script finished.")
