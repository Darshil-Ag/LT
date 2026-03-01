# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies (OpenCV requirements and Tesseract OCR)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code
COPY . .

# Create the upload and output directories so they exist
RUN mkdir -p uploads outputs

# Tell the server how to run your app using Gunicorn (production server)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
