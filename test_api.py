"""
Antigravity - Suíte de Testes Automatizados da API (Pytest + FastAPI TestClient)
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    """Fixture que inicializa o TestClient disparando o Lifespan (startup/shutdown)."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """Testa o endpoint de Health Check essencial para o Render."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["modelo_carregado"] is True
    assert data["versao"] == "1.0.0"
    assert "timestamp" in data
    assert "ambiente" in data


def test_transacao_aprovada(client):
    """Testa uma transação com comportamento padrão e baixo risco."""
    payload = {
        "id_transacao": "tx-test-legit-001",
        "id_usuario": "usr-test-101",
        "valor": 120.00,
        "hora_transacao": 15,
        "tempo_desde_ultima_transacao": 43200.0,
        "distancia_localizacao_km": 5.0,
        "score_dispositivo": 0.95,
        "tipo_transacao": "PIX"
    }
    response = client.post("/v1/analisar-fraude", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id_transacao"] == "tx-test-legit-001"
    assert data["status"] == "APROVADA"
    assert data["probabilidade_fraude"] < 0.30
    assert data["score_risco"] < 30.0
    assert data["latencia_ms"] > 0
    assert "padrões habituais" in data["motivo"]


def test_transacao_bloqueada_ml(client):
    """Testa detecção estocástica de fraude via Random Forest (sem disparar hard-rule)."""
    payload = {
        "id_transacao": "tx-test-fraud-002",
        "id_usuario": "usr-test-999",
        "valor": 15000.00,
        "hora_transacao": 3,
        "tempo_desde_ultima_transacao": 20.0,
        "distancia_localizacao_km": 12.0,
        "score_dispositivo": 0.03,
        "tipo_transacao": "PIX"
    }
    response = client.post("/v1/analisar-fraude", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id_transacao"] == "tx-test-fraud-002"
    assert data["status"] == "BLOQUEADA"
    assert data["probabilidade_fraude"] >= 0.80
    assert data["score_risco"] >= 80.0
    assert "classificador" in data["motivo"] or "crítico" in data["motivo"]


def test_viagem_impossivel_hard_rule(client):
    """Testa regra determinística de Defesa em Profundidade (viagem impossível)."""
    payload = {
        "id_transacao": "tx-test-hard-003",
        "id_usuario": "usr-test-202",
        "valor": 50.00,
        "hora_transacao": 14,
        "tempo_desde_ultima_transacao": 180.0,
        "distancia_localizacao_km": 2800.0,
        "score_dispositivo": 0.99,
        "tipo_transacao": "CARTAO_CREDITO"
    }
    response = client.post("/v1/analisar-fraude", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "BLOQUEADA"
    assert data["score_risco"] == 100.0
    assert "Viagem Impossível" in data["motivo"]


def test_validacao_pydantic_valor_invalido(client):
    """Testa validação de valor não positivo (<= 0)."""
    payload = {
        "id_transacao": "tx-test-invalid-val",
        "id_usuario": "usr-test-303",
        "valor": -10.00,
        "hora_transacao": 12,
        "tempo_desde_ultima_transacao": 100.0,
        "distancia_localizacao_km": 1.0,
        "score_dispositivo": 0.80,
        "tipo_transacao": "PIX"
    }
    response = client.post("/v1/analisar-fraude", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "erro" in data
    assert any("valor" in err["campo"] for err in data["detalhes"])


def test_validacao_pydantic_hora_invalida(client):
    """Testa validação de hora fora do intervalo [0, 23]."""
    payload = {
        "id_transacao": "tx-test-invalid-hora",
        "id_usuario": "usr-test-303",
        "valor": 100.00,
        "hora_transacao": 25,
        "tempo_desde_ultima_transacao": 100.0,
        "distancia_localizacao_km": 1.0,
        "score_dispositivo": 0.80,
        "tipo_transacao": "PIX"
    }
    response = client.post("/v1/analisar-fraude", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("hora_transacao" in err["campo"] for err in data["detalhes"])


def test_validacao_pydantic_score_dispositivo_invalido(client):
    """Testa validação de score fora de [0.0, 1.0]."""
    payload = {
        "id_transacao": "tx-test-invalid-score",
        "id_usuario": "usr-test-303",
        "valor": 100.00,
        "hora_transacao": 12,
        "tempo_desde_ultima_transacao": 100.0,
        "distancia_localizacao_km": 1.0,
        "score_dispositivo": 1.5,
        "tipo_transacao": "PIX"
    }
    response = client.post("/v1/analisar-fraude", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("score_dispositivo" in err["campo"] for err in data["detalhes"])


def test_validacao_pydantic_tipo_transacao_invalido(client):
    """Testa rejeição de canal/tipo não suportado pelo enum."""
    payload = {
        "id_transacao": "tx-test-invalid-tipo",
        "id_usuario": "usr-test-303",
        "valor": 100.00,
        "hora_transacao": 12,
        "tempo_desde_ultima_transacao": 100.0,
        "distancia_localizacao_km": 1.0,
        "score_dispositivo": 0.80,
        "tipo_transacao": "CRIPTOMOEDA"
    }
    response = client.post("/v1/analisar-fraude", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert any("tipo_transacao" in err["campo"] for err in data["detalhes"])
