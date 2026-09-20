# ==============================================================================
# Antigravity - Dockerfile de Produção Otimizado (Multi-Stage / Lean)
# ==============================================================================
FROM python:3.11-slim

# Metadados
LABEL maintainer="Antigravity Team"
LABEL description="API de Alta Performance para Detecção de Fraudes Financeiras"

# Evita geração de arquivos .pyc e força flush imediato do stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Instala dependências do sistema necessárias para compilação (se houver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalação das dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte da aplicação
COPY . .

# Treina o modelo durante o build para garantir que fraud_model.pkl esteja presente
RUN python train_model.py

# Criação de usuário não-root para segurança corporativa
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Exposição da porta
EXPOSE 8000

# Verificação periódica de saúde (Docker Healthcheck)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Comando de inicialização do servidor Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
