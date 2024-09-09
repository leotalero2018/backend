FROM python:3.11

RUN apt update \
 && apt install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
 && apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords

COPY app.py app.py
COPY Ejemplos.pdf Ejemplos.pdf
COPY tls.crt tls.crt
COPY huggingface_hub_batched.py huggingface_hub_batched.py
COPY trt_llm.py trt_llm.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8109"]