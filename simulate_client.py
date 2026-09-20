"""
Antigravity - Simulador de Transações Financeiras em Tempo Real
Script de demonstração e teste de carga/latência da API.
"""

import sys
import time
import requests

DEFAULT_URL = "http://localhost:8000"

CENARIOS = [
    {
        "nome": "Café / Almoço Comercial Habitual",
        "payload": {
            "id_transacao": "tx-sim-001",
            "id_usuario": "usr-pedro-01",
            "valor": 34.90,
            "hora_transacao": 12,
            "tempo_desde_ultima_transacao": 14400.0,
            "distancia_localizacao_km": 1.5,
            "score_dispositivo": 0.98,
            "tipo_transacao": "PIX"
        }
    },
    {
        "nome": "Supermercado Fim de Tarde",
        "payload": {
            "id_transacao": "tx-sim-002",
            "id_usuario": "usr-pedro-01",
            "valor": 289.40,
            "hora_transacao": 18,
            "tempo_desde_ultima_transacao": 21600.0,
            "distancia_localizacao_km": 4.2,
            "score_dispositivo": 0.95,
            "tipo_transacao": "CARTAO_CREDITO"
        }
    },
    {
        "nome": "TED Comercial de Fornecedor",
        "payload": {
            "id_transacao": "tx-sim-003",
            "id_usuario": "usr-empresa-55",
            "valor": 2500.00,
            "hora_transacao": 15,
            "tempo_desde_ultima_transacao": 86400.0,
            "distancia_localizacao_km": 8.0,
            "score_dispositivo": 0.90,
            "tipo_transacao": "TED"
        }
    },
    {
        "nome": "Dispositivo Desconhecido em Horário Atípico",
        "payload": {
            "id_transacao": "tx-sim-004",
            "id_usuario": "usr-lucas-88",
            "valor": 1200.00,
            "hora_transacao": 23,
            "tempo_desde_ultima_transacao": 1200.0,
            "distancia_localizacao_km": 85.0,
            "score_dispositivo": 0.45,
            "tipo_transacao": "PIX"
        }
    },
    {
        "nome": "Invasão de Conta (ATO) - Madrugada & Emulador",
        "payload": {
            "id_transacao": "tx-sim-005",
            "id_usuario": "usr-maria-19",
            "valor": 14850.00,
            "hora_transacao": 3,
            "tempo_desde_ultima_transacao": 15.0,
            "distancia_localizacao_km": 350.0,
            "score_dispositivo": 0.04,
            "tipo_transacao": "PIX"
        }
    },
    {
        "nome": "Viagem Impossível (2.900 km em 2 minutos)",
        "payload": {
            "id_transacao": "tx-sim-006",
            "id_usuario": "usr-carlos-42",
            "valor": 180.00,
            "hora_transacao": 14,
            "tempo_desde_ultima_transacao": 120.0,
            "distancia_localizacao_km": 2900.0,
            "score_dispositivo": 0.92,
            "tipo_transacao": "CARTAO_CREDITO"
        }
    },
    {
        "nome": "Ataque de Velocidade / Rajada (Burst)",
        "payload": {
            "id_transacao": "tx-sim-007",
            "id_usuario": "usr-alvo-77",
            "valor": 4999.00,
            "hora_transacao": 4,
            "tempo_desde_ultima_transacao": 8.0,
            "distancia_localizacao_km": 120.0,
            "score_dispositivo": 0.08,
            "tipo_transacao": "PIX"
        }
    }
]


def executar_simulacao(base_url: str = DEFAULT_URL):
    print("=" * 90)
    print(f"🚀 ANTIGRAVITY ENGINE - SIMULAÇÃO DE TRANSAÇÕES EM TEMPO REAL")
    print(f"Alvo: {base_url}")
    print("=" * 90)

    # 1. Health Check
    try:
        r_health = requests.get(f"{base_url}/", timeout=5)
        if r_health.status_code == 200 and r_health.json().get("status") == "healthy":
            print(f"✅ Motor Online | Versão: {r_health.json().get('versao')} | Modelo Ativo: {r_health.json().get('modelo_carregado')}\n")
        else:
            print(f"⚠️ Alerta: API respondeu com status inesperado: {r_health.text}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Erro de Conexão: Não foi possível conectar a {base_url}.")
        print("Certifique-se de que a API está rodando: uvicorn main:app --reload")
        sys.exit(1)

    # 2. Execução dos Cenários
    endpoint = f"{base_url}/v1/analisar-fraude"
    icones = {
        "APROVADA": "🟢",
        "EM_ANALISE": "🟡",
        "BLOQUEADA": "🔴"
    }

    print(f"{'Cenário':<40} | {'Status':<14} | {'Score':<8} | {'Prob Fraude':<12} | {'Latência':<10}")
    print("-" * 95)

    for c in CENARIOS:
        nome = c["nome"]
        payload = c["payload"]
        try:
            t0 = time.perf_counter()
            resp = requests.post(endpoint, json=payload, timeout=5)
            t_total = (time.perf_counter() - t0) * 1000

            if resp.status_code == 200:
                data = resp.json()
                st = data["status"]
                score = f"{data['score_risco']:.1f}"
                prob = f"{data['probabilidade_fraude']:.4f}"
                lat = f"{data['latencia_ms']:.2f}ms"
                ic = icones.get(st, "⚪")
                print(f"{nome:<40} | {ic} {st:<11} | {score:<8} | {prob:<12} | {lat:<10}")
            else:
                print(f"{nome:<40} | ❌ ERRO {resp.status_code}: {resp.text[:30]}")
        except Exception as e:
            print(f"{nome:<40} | ❌ FALHA: {e}")

    print("=" * 95)
    print("Simulação concluída com sucesso.\n")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    executar_simulacao(url)
