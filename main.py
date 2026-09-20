"""
Antigravity - API de Detecção de Fraudes Financeiras em Alta Performance
Framework: FastAPI + Uvicorn
Arquitetura: Clean Architecture & Microserviço Preditivo para Deploy no Render
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from enum import Enum
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# 1. Configuração de Logging Estruturado
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [AntigravityAPI] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("antigravity_api")

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "fraud_model.pkl"))
DASHBOARD_HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")

# ---------------------------------------------------------------------------
# 2. Ciclo de Vida da Aplicação (FastAPI Lifespan)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciador de ciclo de vida do FastAPI.
    Carrega o modelo serializado na memória durante a inicialização (startup)
    e garante que o modelo esteja pronto antes de aceitar tráfego.
    """
    logger.info("Iniciando processo de inicialização do serviço...")
    if os.path.exists(MODEL_PATH):
        try:
            app.state.model = joblib.load(MODEL_PATH)
            logger.info(f"Modelo de fraude carregado com sucesso a partir de '{MODEL_PATH}'.")
        except Exception as exc:
            logger.error(f"Falha crítica ao carregar o modelo de '{MODEL_PATH}': {exc}", exc_info=True)
            app.state.model = None
    else:
        logger.warning(
            f"Arquivo do modelo não encontrado em '{MODEL_PATH}'. "
            "Execute 'python train_model.py' para gerar o modelo antes de receber chamadas de predição."
        )
        app.state.model = None

    yield

    logger.info("Encerrando Antigravity API e liberando recursos...")
    app.state.model = None


# ---------------------------------------------------------------------------
# 3. Inicialização do App FastAPI e CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Antigravity - Fraud Detection Engine",
    description=(
        "API de missão crítica para avaliação de risco e detecção de fraudes em transações financeiras em tempo real. "
        "Desenvolvida com FastAPI, Scikit-Learn e arquitetada para alta disponibilidade no Render."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuração de CORS para viabilizar integração com Dashboards / SPAs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção estrita, restringir aos domínios autorizados
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 4. Esquemas de Dados e Validação Rigorosa (Pydantic v2)
# ---------------------------------------------------------------------------
class TipoTransacaoEnum(str, Enum):
    PIX = "PIX"
    CARTAO_CREDITO = "CARTAO_CREDITO"
    BOLETO = "BOLETO"
    TED = "TED"


class TipoConexaoEnum(str, Enum):
    RESIDENCIAL = "RESIDENCIAL"
    MOVEL_4G_5G = "MOVEL_4G_5G"
    VPN_PROXY = "VPN_PROXY"
    TOR = "TOR"


class BeneficiarioNovoEnum(str, Enum):
    NAO = "NAO"
    SIM = "SIM"


class StatusDecisaoEnum(str, Enum):
    BLOQUEADA = "BLOQUEADA"
    EM_ANALISE = "EM_ANALISE"
    APROVADA = "APROVADA"


class TransacaoInput(BaseModel):
    id_transacao: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Identificador único global da transação (UUID ou hash)",
        examples=["tx-98213840-77a2-41bf-8ef4-9d58a0eefc3a"]
    )
    id_usuario: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Identificador único do cliente solicitante",
        examples=["usr-104928"]
    )
    valor: float = Field(
        ...,
        gt=0.0,
        description="Valor nominal da transação financeira em Reais (BRL)",
        examples=[350.50]
    )
    hora_transacao: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hora do dia em que a transação foi originada (0 a 23)",
        examples=[14]
    )
    tempo_desde_ultima_transacao: float = Field(
        ...,
        ge=0.0,
        description="Tempo decorrido (em segundos) desde a última operação deste usuário",
        examples=[3600.0]
    )
    distancia_localizacao_km: float = Field(
        ...,
        ge=0.0,
        description="Distância estimada (em km) entre a localização da transação e o domicílio habitual",
        examples=[12.5]
    )
    score_dispositivo: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de reputação e autenticidade do dispositivo (0.0 = emulador/novo, 1.0 = confiável)",
        examples=[0.95]
    )
    tipo_transacao: TipoTransacaoEnum = Field(
        default=TipoTransacaoEnum.PIX,
        description="Canal ou meio de pagamento utilizado",
        examples=["PIX"]
    )
    # Parâmetros Avançados de Risco Bancário (Enterprise)
    idade_conta_meses: float = Field(
        default=24.0,
        ge=0.0,
        description="Idade da conta bancária do titular em meses",
        examples=[36.0]
    )
    tentativas_falhas_24h: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Contagem de tentativas incorretas de senha/biometria nas últimas 24h",
        examples=[0]
    )
    score_credito_bureau: float = Field(
        default=750.0,
        ge=0.0,
        le=1000.0,
        description="Score de crédito nos órgãos regulatórios/bureaus (0 a 1000)",
        examples=[780.0]
    )
    beneficiario_novo: BeneficiarioNovoEnum = Field(
        default=BeneficiarioNovoEnum.NAO,
        description="Indica se o favorecido/chave PIX foi cadastrado há menos de 24 horas",
        examples=["NAO"]
    )
    tipo_conexao: TipoConexaoEnum = Field(
        default=TipoConexaoEnum.RESIDENCIAL,
        description="Classificação da rede de conexão (Residencial, Móvel, VPN, Tor)",
        examples=["RESIDENCIAL"]
    )

    @field_validator("valor")
    @classmethod
    def validar_valor_maximo(cls, v: float) -> float:
        if v > 10_000_000.0:
            raise ValueError("Valor da transação excede o teto regulatório permitido de R$ 10.000.000,00.")
        return round(v, 2)


class AnaliseFraudeResponse(BaseModel):
    id_transacao: str
    id_usuario: str
    status: StatusDecisaoEnum
    probabilidade_fraude: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade estimada de fraude entre 0.0 e 1.0"
    )
    score_risco: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Score normalizado de risco de 0 a 100"
    )
    motivo: str
    fatores_risco: list[str] = Field(
        default_factory=list,
        description="Indicadores técnicos de risco detectados para auditoria de compliance"
    )
    latencia_ms: float
    data_processamento: datetime


class HealthCheckResponse(BaseModel):
    status: str
    modelo_carregado: bool
    versao: str
    timestamp: datetime
    ambiente: str


# ---------------------------------------------------------------------------
# 5. Tratamento de Exceções Centralizado
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Retorna erros de validação Pydantic de maneira limpa e orientada ao cliente."""
    erros = [
        {"campo": " -> ".join(str(loc) for loc in err["loc"]), "mensagem": err["msg"]}
        for err in exc.errors()
    ]
    logger.warning(f"Payload inválido recebido em {request.url.path}: {erros}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "erro": "Dados de transação inválidos",
            "detalhes": erros,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Tratamento preventivo de falhas não previstas para manter a estabilidade."""
    logger.error(f"Erro inesperado no endpoint {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "erro": "Erro interno no motor antifraude",
            "mensagem": "Ocorreu uma inconsistência no processamento da análise. Notifique a equipe de engenharia.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# ---------------------------------------------------------------------------
# 6. Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/",
    response_model=HealthCheckResponse,
    tags=["Monitoramento"],
    summary="Health Check para monitoramento e orquestradores (Render)",
    description="Endpoint essencial de verificação de disponibilidade, utilizado pelo Render para liveness e readiness checks."
)
async def health_check():
    modelo_ativo = hasattr(app.state, "model") and app.state.model is not None
    return HealthCheckResponse(
        status="healthy" if modelo_ativo else "degraded",
        modelo_carregado=modelo_ativo,
        versao=app.version,
        timestamp=datetime.now(timezone.utc),
        ambiente=os.getenv("ENVIRONMENT", "production")
    )


@app.post(
    "/v1/analisar-fraude",
    response_model=AnaliseFraudeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Detecção de Fraude"],
    summary="Analisa e avalia o risco de fraude de uma transação",
    description="Calcula a inferência do modelo Random Forest, aplica matriz de risco e regras determinísticas de negócio."
)
async def analisar_fraude(transacao: TransacaoInput):
    inicio_tempo = time.perf_counter()

    # Validação da disponibilidade do modelo de Machine Learning
    if not hasattr(app.state, "model") or app.state.model is None:
        logger.critical("Tentativa de inferência sem o modelo fraud_model.pkl carregado.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O motor antifraude está temporariamente indisponível. O modelo preditivo não foi carregado."
        )

    # 1. Regras Determinísticas de Defesa em Profundidade (Hard Rules)
    fatores: list[str] = []

    # Heurística 1: Viagem Impossível
    if transacao.distancia_localizacao_km >= 1500.0 and transacao.tempo_desde_ultima_transacao < 600.0:
        latencia = round((time.perf_counter() - inicio_tempo) * 1000, 2)
        fatores.append("Deslocamento geográfico fisicamente impossível (velocidade > 9.000 km/h).")
        logger.warning(f"Transação {transacao.id_transacao} BLOQUEADA via Hard-Rule: Viagem Impossível.")
        return AnaliseFraudeResponse(
            id_transacao=transacao.id_transacao,
            id_usuario=transacao.id_usuario,
            status=StatusDecisaoEnum.BLOQUEADA,
            probabilidade_fraude=0.9999,
            score_risco=100.0,
            motivo="Bloqueio determinístico de segurança: Deslocamento físico incompatível com o intervalo de tempo (Viagem Impossível).",
            fatores_risco=fatores,
            latencia_ms=latencia,
            data_processamento=datetime.now(timezone.utc)
        )

    # Heurística 2: Conexão TOR com Alto Valor
    if transacao.tipo_conexao == TipoConexaoEnum.TOR and transacao.valor >= 2000.0:
        latencia = round((time.perf_counter() - inicio_tempo) * 1000, 2)
        fatores.append("Transação de alto valor originada via rede anonimizada TOR.")
        return AnaliseFraudeResponse(
            id_transacao=transacao.id_transacao,
            id_usuario=transacao.id_usuario,
            status=StatusDecisaoEnum.BLOQUEADA,
            probabilidade_fraude=0.9990,
            score_risco=99.9,
            motivo="Bloqueio de conformidade: Acesso originado por nó de saída TOR em operação acima do limite cautelar.",
            fatores_risco=fatores,
            latencia_ms=latencia,
            data_processamento=datetime.now(timezone.utc)
        )

    # 2. Preparação do DataFrame de entrada alinhado ao Pipeline do Scikit-Learn
    dados_inferencia = pd.DataFrame([{
        "valor": transacao.valor,
        "hora_transacao": transacao.hora_transacao,
        "tempo_desde_ultima_transacao": transacao.tempo_desde_ultima_transacao,
        "distancia_localizacao_km": transacao.distancia_localizacao_km,
        "score_dispositivo": transacao.score_dispositivo,
        "idade_conta_meses": transacao.idade_conta_meses,
        "tentativas_falhas_24h": transacao.tentativas_falhas_24h,
        "score_credito_bureau": transacao.score_credito_bureau,
        "beneficiario_novo": transacao.beneficiario_novo.value,
        "tipo_conexao": transacao.tipo_conexao.value,
        "tipo_transacao": transacao.tipo_transacao.value
    }])

    # 3. Execução da Inferência no Pipeline Treinado
    try:
        probabilidades = app.state.model.predict_proba(dados_inferencia)
        probabilidade_fraude = float(probabilidades[0][1])
    except Exception as exc:
        logger.error(f"Falha na inferência para a transação {transacao.id_transacao}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Falha interna durante o cálculo probabilístico da transação."
        )

    # 4. Atribuição de Fatores de Risco para Auditoria
    if transacao.distancia_localizacao_km >= 300.0:
        fatores.append(f"Distância atípica do domicílio habitual ({transacao.distancia_localizacao_km:.0f} km)")
    if transacao.tipo_conexao in (TipoConexaoEnum.TOR, TipoConexaoEnum.VPN_PROXY):
        fatores.append(f"Conexão mascarada ({transacao.tipo_conexao.value})")
    if transacao.tentativas_falhas_24h >= 1:
        fatores.append(f"{transacao.tentativas_falhas_24h} falha(s) de autenticação nas últimas 24h")
    if transacao.beneficiario_novo == BeneficiarioNovoEnum.SIM:
        fatores.append("Favorecido / chave PIX cadastrado há menos de 24h")
    if transacao.idade_conta_meses <= 3.0:
        fatores.append(f"Conta recente (< 3 meses)")
    if transacao.score_dispositivo <= 0.35:
        fatores.append("Hardware com baixa confiabilidade / suspeita de emulador")
    if transacao.hora_transacao in [0, 1, 2, 3, 4, 5]:
        fatores.append("Operação na madrugada (janela de risco bancário)")
    if transacao.score_credito_bureau < 450:
        fatores.append(f"Score bureau restritivo ({int(transacao.score_credito_bureau)}/1000)")

    # 5. Aplicação das Regras de Negócio e Matriz de Decisão
    score_risco = round(probabilidade_fraude * 100, 2)
    prob_formatada = round(probabilidade_fraude, 4)

    if probabilidade_fraude >= 0.80:
        decisao = StatusDecisaoEnum.BLOQUEADA
        motivo = f"Risco crítico ({score_risco}/100). Alta probabilidade de invasão de conta ou golpe financeiro."
    elif probabilidade_fraude >= 0.30:
        decisao = StatusDecisaoEnum.EM_ANALISE
        motivo = f"Risco moderado ({score_risco}/100). Indicadores comportamentais atípicos exigem validação (2FA)."
    else:
        decisao = StatusDecisaoEnum.APROVADA
        motivo = f"Transação segura ({score_risco}/100). Operação em conformidade com o histórico transacional."

    latencia = round((time.perf_counter() - inicio_tempo) * 1000, 2)

    logger.info(
        f"Transação {transacao.id_transacao} avaliada: Decisão={decisao.value} | "
        f"Prob={prob_formatada} | Latência={latencia}ms"
    )

    return AnaliseFraudeResponse(
        id_transacao=transacao.id_transacao,
        id_usuario=transacao.id_usuario,
        status=decisao,
        probabilidade_fraude=prob_formatada,
        score_risco=score_risco,
        motivo=motivo,
        fatores_risco=fatores,
        latencia_ms=latencia,
        data_processamento=datetime.now(timezone.utc)
    )


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    tags=["Interface Visual"],
    summary="Painel Interativo de Gestão Antifraude",
    description="Interface amigável para analistas de risco e finanças realizarem análises em tempo real."
)
@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Entrega a interface visual moderna (Fintech Dashboard) para uso do time financeiro."""
    if os.path.exists(DASHBOARD_HTML_PATH):
        with open(DASHBOARD_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h2>Painel em atualização. Arquivo dashboard.html não encontrado.</h2>",
        status_code=404
    )


# ---------------------------------------------------------------------------
# Ponto de Entrada para Execução Direta (Local Dev)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    porta = int(os.getenv("PORT", 8000))
    logger.info(f"Subindo servidor de desenvolvimento local na porta {porta}...")
    uvicorn.run("main:app", host="0.0.0.0", port=porta, reload=True)
