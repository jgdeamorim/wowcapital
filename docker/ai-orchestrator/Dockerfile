FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    pkg-config \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-ai.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-ai.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV OPENAI_API_KEY=""
ENV TRADING_MODE=demo
ENV LOG_LEVEL=INFO

# Command to run the application
CMD ["python", "-m", "src.ai_orchestrator.main"]