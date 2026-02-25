FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api:app --host 127.0.0.1 --port 8000 & streamlit run chatbot.py --server.enableCORS false --server.enableXsrfProtection false --server.address 0.0.0.0 --server.port 7860 --server.headless true"]
