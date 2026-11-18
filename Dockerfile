# CyberRAG: Graph-Augmented RAG with NLI Verification and Temporal Contradiction Resolution
FROM python:3.10-slim

# Prevent Python from writing .pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository contents
COPY . .

# Expose Workbench (8080) and Streamlit (8501) ports
EXPOSE 8080 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/api/health || exit 1

# Launch CyberRAG Analyst Workbench by default
ENTRYPOINT ["python", "demo/server.py"]
