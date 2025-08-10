import os
import requests
import zipfile
import shutil
from pathlib import Path

def download_and_extract(url: str, extract_to: Path):
    zip_path = extract_to / "dataset.zip"

    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from {url} ...")
    response = requests.get(url, stream=True)
    total_length = response.headers.get('content-length')

    with open(zip_path, "wb") as f:
        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                print(f"\r[{'=' * done}{' ' * (50 - done)}] {dl*100/total_length:.2f}%", end='')

    print("\nExtracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_path)
    print("Download and extraction complete.")

def organize_dataset(source_dir: Path, target_dir: Path):
    cats_dst = target_dir / "data/train/cats"
    dogs_dst = target_dir / "data/train/dogs"

    cats_dst.mkdir(parents=True, exist_ok=True)
    dogs_dst.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {source_dir} for images...")

    # automatic sorting for cats and dogs images
    for img_path in source_dir.rglob("*.*"):
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            fname = img_path.name.lower()
            if "cat" in fname:
                shutil.move(str(img_path), cats_dst / img_path.name)
            elif "dog" in fname:
                shutil.move(str(img_path), dogs_dst / img_path.name)
            else:
                # skip files without cat or dog in name
                pass

    print("Dataset organized into 'data/train/cats' and 'data/train/dogs'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and organize Cats vs Dogs dataset.")
    parser.add_argument("--url", type=str,
                        default="https://github.com/microsoft/ML-Images/archive/refs/heads/master.zip", # default github dataset repo link
                        help="URL of the ZIP dataset to download.")
    parser.add_argument("--extract_to", type=str, default="dataset_raw",
                        help="Temporary folder to extract dataset.")
    parser.add_argument("--target_dir", type=str, default=".",
                        help="Target root directory for data/train folders.")

    args = parser.parse_args()

    extract_path = Path(args.extract_to)
    target_path = Path(args.target_dir)

    download_and_extract(args.url, extract_path)

    # We pick the first folder inside extract_path as the source_dir:
    extracted_subdirs = [p for p in extract_path.iterdir() if p.is_dir()]
    if not extracted_subdirs:
        print(f"No extracted folder found inside {extract_path}")
        source_dir = extract_path
    else:
        source_dir = extracted_subdirs[0]

    organize_dataset(source_dir, target_path)

    # Cleanup extracted raw files
    shutil.rmtree(extract_path)
