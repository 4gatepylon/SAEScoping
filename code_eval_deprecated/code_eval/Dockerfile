# Use the official Python image (Debian Bookworm-based)
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies
# -> This is recommended by Claude
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY code_runner_server.py .
COPY code_runner_utils.py .
COPY code_runner_schemas.py .


# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Meant to be like:
# ```
# services:
#   code-execution-server:
#     build:
#       context: .
#       dockerfile: Dockerfile
#     container_name: code-execution-server
#     ports:
#       - "8000:8000"
#     environment:
#       - PYTHONUNBUFFERED=1
#       - PYTHONDONTWRITEBYTECODE=1
#     restart: no
# ```
# (but uses python to run many)

# Run the server
CMD ["python", "code_runner_server.py"]