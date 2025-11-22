# Clean PII Images API

This repository provides a Flask API for **removing PII from images** using a YOLO-based detection model. You can run it locally via **Docker** and clean your images easily.

---

## **Features**

- Upload images via API
- Detect PII regions
- Black out PII areas and return the cleaned image
- Batch processing supported via Python scripts

---

## **Prerequisites**

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- Optional: Python 3.10+ (if running scripts outside Docker)

---

## **1. Clone the repository**

```bash
git clone https://github.com/<your-repo>/clean-pii-sti.git
cd clean-pii-sti
```

## **2. Pull the Docker Image**
```bash
docker pull musinguzidenis/clean-pii-sti:latest
```

## **3. Run the Docker Image**
```bash
docker run -p 8032:8032 musinguzidenis/clean-pii-sti:latest
```

## **4. Create a text file with paths to the X-ray Images**
The text file should look like this.
```
path/image-1.jpg
path/image-2.jpg
```

## **5. Run the Python script to clean the Images**
```
python clean_images.py --input_list "path-to-images.txt" --output_dir path-to-output-dir --api_url http://localhost:8082/predict
```