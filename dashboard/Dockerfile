FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache busting for production builds - set via: docker build --build-arg CACHEBUST=$(date +%s)
ARG CACHEBUST=1
RUN echo "Build timestamp: ${CACHEBUST}"

# Copy app
COPY main.py .

# Create data directory
RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
