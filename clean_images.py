import os
import argparse
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Batch clean images through Flask API")
    parser.add_argument(
        "--input_list",
        type=str,
        required=True,
        help="Path to text file containing image paths, one per line",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save cleaned images",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8082/predict",
        help="Flask endpoint URL",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if missing
    os.makedirs(args.output_dir, exist_ok=True)

    # Read image paths
    with open(args.input_list, "r") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"[WARNING] File not found: {img_path}")
            continue

        print(f"[INFO] Processing: {img_path}")

        # Send image to API
        with open(img_path, "rb") as file:
            response = requests.post(args.api_url, files={"image": file})

        if response.status_code != 200:
            print(f"[ERROR] API error for {img_path}: {response.text}")
            continue

        # Save cleaned image
        filename = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"[OK] Saved → {output_path}")


if __name__ == "__main__":
    main()
