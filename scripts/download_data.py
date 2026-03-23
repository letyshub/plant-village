"""Download PlantVillage dataset from Kaggle and copy it into data/raw/plantvillage."""

import shutil
from pathlib import Path

import kagglehub

DEST = Path("data/raw/plantvillage")


def main() -> None:
    print("Downloading PlantVillage dataset via kagglehub...")
    path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    print(f"Downloaded to cache: {path}")

    # kagglehub may wrap the classes in an extra subdirectory.
    # Find the level that contains the actual class folders (names with "___").
    source = Path(path)
    subdirs = [d for d in source.iterdir() if d.is_dir()]
    if subdirs and not any("___" in d.name for d in subdirs):
        # One level too high — step into the single subfolder
        source = subdirs[0]
        print(f"Stepping into subfolder: {source.name}")

    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, DEST, dirs_exist_ok=True)
    print(f"Dataset ready at: {DEST.resolve()}")

    classes = [d.name for d in sorted(DEST.iterdir()) if d.is_dir()]
    print(f"Found {len(classes)} classes: {classes[:5]} ...")


if __name__ == "__main__":
    main()
