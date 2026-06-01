# ──────────────────────────────────────────────
#  ClusterLens — Dockerfile
#  Aplicação de inferência com modelo hierárquico
# ──────────────────────────────────────────────

FROM python:3.9-slim

# Metadados
LABEL maintainer="ClusterLens"
LABEL description="Aplicação web para classificação de texto por clustering hierárquico"

# Diretório de trabalho
WORKDIR /app

# Copiar dependências primeiro (aproveita cache de camadas)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar modelos treinados
COPY models/ ./models/

# Copiar código da aplicação
COPY app.py .
COPY templates/ ./templates/

# Variáveis de ambiente
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expor porta
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')"

# Iniciar com Gunicorn (servidor WSGI de produção)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app:app"]
