# Immagine che serve UNA versione del modello, esportata prima dal registry:
#   python -m scripts.esporta_modello            (crea serving/model/)
#   docker build -t har-classificatore:v1 serving/
#   docker run --rm -p 8080:8080 har-classificatore:v1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/model

WORKDIR /app

# Prima le dipendenze (strato che cambia di rado, quindi resta in cache), poi codice e modello.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY model/ model/

# Niente root dentro il container.
RUN useradd --create-home --uid 10001 servizio
USER servizio

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
