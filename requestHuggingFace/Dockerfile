FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# The path exists because of the docker compose file
COPY nginx.conf /etc/nginx/nginx.conf

# creates home directory /home/appuser setting the user id to 1000

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1


# Run the FastAPI app called app from main.py using Uvicorn on port 8000, accepting external connections, using one worker.”
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]