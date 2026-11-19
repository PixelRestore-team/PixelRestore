
FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code and required models
COPY app/ ./app/
COPY src/ ./src/
COPY models/trn/trn_generator.keras ./models/trn/trn_generator.keras
COPY srn_unet_model_128.keras ./
COPY run_web.py .

# Cloud Run uses port 8080 by default
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
