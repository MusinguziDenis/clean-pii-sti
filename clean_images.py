import argparse
from http import HTTPStatus
from pathlib import Path

import requests

REQUEST_TIMEOUT_SECONDS = 30


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Batch clean images through Flask API",
    )
    parser.add_argument(
        "--input_list",
        type=Path,
        required=True,
        help="Path to text file containing image paths, one per line",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
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


def main() -> None:
    """Read input paths, call the API, and write cleaned images."""
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_list: Path = args.input_list
    with input_list.open(encoding="utf-8") as handle:
        image_paths = [Path(line.strip()) for line in handle if line.strip()]

    for img_path in image_paths:
        if not img_path.exists():
            print(f"[WARNING] File not found: {img_path}")
            continue

        print(f"[INFO] Processing: {img_path}")

        with img_path.open("rb") as file:
            response = requests.post(
                args.api_url,
                files={"image": file},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        if response.status_code != HTTPStatus.OK:
            print(f"[ERROR] API error for {img_path}: {response.text}")
            continue

        output_path = output_dir / img_path.name
        with output_path.open("wb") as handle:
            handle.write(response.content)

        print(f"[OK] Saved -> {output_path}")


if __name__ == "__main__":
    main()
