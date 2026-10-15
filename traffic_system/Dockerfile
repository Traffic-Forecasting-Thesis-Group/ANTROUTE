FROM python:3.12-slim

WORKDIR /app

# opencv-python (requirements.txt) is the desktop build and links against X11 at import.
# Route planning reaches it through src/vision/frame_extractor.py, so without these libraries
# the whole /routes/plan chain dies on "libxcb.so.1: cannot open shared object file".
# Switching requirements.txt to opencv-python-headless would drop them entirely -- nothing in
# this codebase calls an OpenCV GUI function -- but that file is shared with the Colab
# notebooks, so the fix is kept to the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
