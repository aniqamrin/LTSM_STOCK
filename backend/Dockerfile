FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . /app/backend/

# Copy pre-computed pipeline outputs
COPY ../results /app/results
COPY ../sentiment_scores /app/sentiment_scores
COPY ../saved_models /app/saved_models
COPY ../processed_data /app/processed_data

WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
