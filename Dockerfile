FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    graphviz \
    libgraphviz-dev \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -v -r requirements.txt

COPY . .

RUN mkdir -p uploads outputs

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]