import os
import shutil
from pathlib import Path

import cv2
import pandas as pd
import yaml
from tqdm import tqdm


def save_dataset_in_yolo(
        image_folder:str,
        preprocessed_df:pd.DataFrame,
        output_dir:str,
        ) -> dict:
    """Convert a dataset into YOLO format and saves it to disk.

    Exlude images without bounding boxes.
    Also generates a dataset.yaml file for YOLO configuration.

    Args:
        image_folder (str): Path to folder containing source images
        preprocessed_df (pd.DataFrame): DataFrame containing annotations
        with columns:
            ['Image Index', 'Finding Label', 'Bbox [x', 'y', 'w', 'h]']
        output_dir (str): Path where the YOLO dataset will be saved
    Returns:
        dict: Statistics about the processed dataset

    """
    # remove output dir
    shutil.rmtree(output_dir, ignore_errors=True)

    # Create output directory
    Path.mkdir(Path(output_dir), exist_ok=False, parents=True)

    # Clear existing directory if it exists
    shutil.rmtree(os.path.join(output_dir, "train"), ignore_errors=True)

    # Remove NEG class images
    pos_df = preprocessed_df[preprocessed_df["Finding Label"] != "NEG"]

    # Get unique positive classes and create mapping (excluding NEG)
    unique_classes = sorted(pos_df["Finding Label"].unique().tolist())
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}

    # Get unique image IDs (excluding images that only have NEG classes)
    image_ids = pos_df["Image Index"].unique()

    # Create directories
    train_dir = os.path.join(output_dir, "train")
    Path.mkdir(Path(os.path.join(train_dir, "labels")), exist_ok=True)
    Path.mkdir(Path(os.path.join(train_dir, "images")), exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for img_name in tqdm(image_ids, desc="Processing images"):
        # Load image
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Unable to read image {img_path} - Skipping.")
            skipped_count += 1
            continue

        imheight, imwidth, _ = img.shape

        # Get annotations for this image (excluding NEG class)
        img_boxes = pos_df[pos_df["Image Index"] == img_name]

        # Create label file
        label_filename = f"{''.join(img_name.split('.')[:-1])}.txt"
        label_path = Path(os.path.join(train_dir, "labels", label_filename))

        with label_path.open("w+") as f:
            for _, row in img_boxes.iterrows():
                # Calculate normalized box dimensions
                x_center = (row["x"] + row["w"]/2) / imwidth
                y_center = (row["y"] + row["h"]/2) / imheight
                width = row["w"] / imwidth
                height = row["h"] / imheight

                # Write YOLO format line
                class_id = class_mapping[row["Finding Label"]]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        # Save image
        cv2.imwrite(os.path.join(train_dir, "images", img_name), img)
        processed_count += 1


    # Save class mapping
    classes_file_path = Path(os.path.join(output_dir, "classes.txt"))
    with classes_file_path.open("w+") as f:
        for class_name, _ in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")

    # Generate dataset.yaml
    yaml_content = {
        "names": {class_mapping[k]: k for k in class_mapping},
        "path": str(Path(output_dir).resolve()),
        "train": "train",
        "val": "train",
    }

    # Save dataset.yaml
    yaml_path = Path(os.path.join(output_dir, "dataset.yaml"))
    with yaml_path.open("w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # Compile statistics
    stats = {
        "total_images_before_filtering": len(
            preprocessed_df["Image Index"].unique()),
        "total_images_after_filtering": len(image_ids),
        "processed": processed_count,
        "skipped": skipped_count,
        "class_mapping": class_mapping,
        "classes": unique_classes,
    }

    print("\nDataset generation complete!")
    print(
        f"Total images before filtering:\
            {stats['total_images_before_filtering']}",
        )
    print(
        f"Total images after removing NEG class:\
              {stats['total_images_after_filtering']}",
    )
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Classes: {', '.join(stats['classes'])}") # type: ignore[arg-type]
    print(f"Dataset YAML file saved to: {yaml_path}")

    return stats


if __name__ == "__main__":

    # Load preprocessed dataset
    preprocessed_df = pd.read_csv("data/lacuna_chest_xray2017_data.csv")

    # Save dataset in YOLO format
    save_dataset_in_yolo(
        image_folder="data/images/",
        preprocessed_df=preprocessed_df,
        output_dir="data/yolo_dataset",
    )
