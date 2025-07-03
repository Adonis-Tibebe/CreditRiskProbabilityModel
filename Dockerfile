# Dockerfile

FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy requirements & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY src/ ./src/
COPY models/ ./models/ 

# Run the API using uvicorn
#CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]