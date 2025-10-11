FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY server.py .

# Create data directory
RUN mkdir -p /data

# Expose HTTP port
EXPOSE 9393

# Run server
CMD ["python", "server.py"]