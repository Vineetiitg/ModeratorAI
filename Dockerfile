FROM python:3.11-slim

# Prevent Python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install dependencies for both services
RUN pip install --no-cache-dir -r ml-service/requirements.txt
RUN pip install --no-cache-dir -r python-frontend/requirements.txt

# Ensure start script is executable
RUN chmod +x start.sh

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Run the unified entrypoint script
CMD ["./start.sh"]
