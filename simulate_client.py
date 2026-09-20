"""
FraudGuard - Simulador de Transações Financeiras em Tempo Real (Enterprise)
Utiliza a biblioteca padrão para envio de cenários com telemetria bancária avançada.
"""

import sys
import time
import json
import urllib.request
import urllib.error

# Suporte a UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_URL = "https://antigravity-fraud-api.onrender.com"

CENARIOS = [
    {
        "nome": "Operação Habitual (Comercial)",
        "payload": {
            "id_transacao": "TX-PROD-001",
            "id_usuario": "USR-PEDRO-01",
            "valor": 150.00,
            "hora_transacao": 14,
            "tempo_desde_ultima_transacao": 18000.0,
            "distancia_localizacao_km": 2.5,
            "score_dispositivo": 0.98,
            "tipo_transacao": "PIX",
            "idade_conta_meses": 36.0,
            "tentativas_falhas_24h": 0,
            "score_credito_bureau": 810.0,
            "beneficiario_novo": "NAO",
            "tipo_conexao": "RESIDENCIAL"
        }
    },
    {
        "nome": "Supermercado Fim de Tarde (Cartão)",
        "payload": {
            "id_transacao": "TX-PROD-002",
            "id_usuario": "USR-PEDRO-01",
            "valor": 340.50,
            "hora_transacao": 18,
            "tempo_desde_ultima_transacao": 21600.0,
            "distancia_localizacao_km": 4.2,
            "score_dispositivo": 0.95,
            "tipo_transacao": "CARTAO_CREDITO",
            "idade_conta_meses": 36.0,
            "tentativas_falhas_24h": 0,
            "score_credito_bureau": 810.0,
            "beneficiario_novo": "NAO",
            "tipo_conexao": "MOVEL_4G_5G"
        }
    },
    {
        "nome": "Transferência Noturna + Novo Favorecido",
        "payload": {
            "id_transacao": "TX-PROD-003",
            "id_usuario": "USR-MARINA-55",
            "valor": 1800.00,
            "hora_transacao": 22,
            "tempo_desde_ultima_transacao": 2700.0,
            "distancia_localizacao_km": 35.0,
            "score_dispositivo": 0.60,
            "tipo_transacao": "TED",
            "idade_conta_meses": 8.0,
            "tentativas_falhas_24h": 1,
            "score_credito_bureau": 550.0,
            "beneficiario_novo": "SIM",
            "tipo_conexao": "MOVEL_4G_5G"
        }
    },
    {
        "nome": "Invasão de Conta (ATO) - Madrugada & VPN",
        "payload": {
            "id_transacao": "TX-PROD-004",
            "id_usuario": "USR-LUCAS-88",
            "valor": 14500.00,
            "hora_transacao": 3,
            "tempo_desde_ultima_transacao": 20.0,
            "distancia_localizacao_km": 180.0,
            "score_dispositivo": 0.05,
            "tipo_transacao": "PIX",
            "idade_conta_meses": 2.0,
            "tentativas_falhas_24h": 3,
            "score_credito_bureau": 390.0,
            "beneficiario_novo": "SIM",
            "tipo_conexao": "VPN_PROXY"
        }
    },
    {
        "nome": "Rede TOR Anonimizada + Alto Valor",
        "payload": {
            "id_transacao": "TX-PROD-005",
            "id_usuario": "USR-SUSPEITO-99",
            "valor": 6200.00,
            "hora_transacao": 2,
            "tempo_desde_ultima_transacao": 15.0,
            "distancia_localizacao_km": 450.0,
            "score_dispositivo": 0.05,
            "tipo_transacao": "PIX",
            "idade_conta_meses": 1.0,
            "tentativas_falhas_24h": 4,
            "score_credito_bureau": 310.0,
            "beneficiario_novo": "SIM",
            "tipo_conexao": "TOR"
        }
    },
    {
        "nome": "Viagem Impossível (2.800 km em 2 min)",
        "payload": {
            "id_transacao": "TX-PROD-006",
            "id_usuario": "USR-CARLOS-42",
            "valor": 350.00,
            "hora_transacao": 15,
            "tempo_desde_ultima_transacao": 120.0,
            "distancia_localizacao_km": 2800.0,
            "score_dispositivo": 0.90,
            "tipo_transacao": "CARTAO_CREDITO",
            "idade_conta_meses": 24.0,
            "tentativas_falhas_24h": 0,
            "score_credito_bureau": 720.0,
            "beneficiario_novo": "NAO",
            "tipo_conexao": "RESIDENCIAL"
        }
    }
]


def executar_simulacao(base_url: str = DEFAULT_URL):
    base_url = base_url.rstrip("/")
    print("=" * 115)
    print("🛡️ FRAUDGUARD RISK DESK - CONSOLE DE TESTES OPERACIONAIS")
    print(f"Alvo: {base_url}")
    print("=" * 115)

    # 1. Health Check
    try:
        req = urllib.request.Request(f"{base_url}/", headers={"User-Agent": "FraudGuardSimulator/2.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data_health = json.loads(resp.read().decode("utf-8"))
            print(f"Motor: ONLINE | Versão: {data_health.get('versao')} | Modelo Ativo: {data_health.get('modelo_carregado')}\n")
    except Exception as exc:
        print(f"Erro de conexão com {base_url}: {exc}")
        sys.exit(1)

    endpoint = f"{base_url}/v1/analisar-fraude"
    icones = {
        "APROVADA": "🟢",
        "EM_ANALISE": "🟡",
        "BLOQUEADA": "🔴"
    }

    print(f"{'Cenário':<42} | {'Decisão':<14} | {'Score':<7} | {'Prob':<7} | {'Latência':<10} | {'Fatores Principais'}")
    print("-" * 125)

    for c in CENARIOS:
        nome = c["nome"]
        payload = c["payload"]
        dados_json = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=dados_json,
            headers={"Content-Type": "application/json", "User-Agent": "FraudGuardSimulator/2.0"}
        )

        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=15) as resp:
                rtt = (time.perf_counter() - t0) * 1000
                data = json.loads(resp.read().decode("utf-8"))
                st = data["status"]
                score = f"{data['score_risco']:.1f}"
                prob = f"{data['probabilidade_fraude']:.4f}"
                lat_srv = f"{data['latencia_ms']:.2f}ms"
                ic = icones.get(st, "⚪")
                fatores = ", ".join(data.get("fatores_risco", [])[:2]) or "Nenhum"
                print(f"{nome:<42} | {ic} {st:<11} | {score:<7} | {prob:<7} | {lat_srv:<10} | {fatores[:45]}")
        except Exception as err:
            print(f"{nome:<42} | ❌ ERRO: {err}")

    print("=" * 125)
    print("Processamento concluído com sucesso.\n")


if __name__ == "__main__":
    url_alvo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    executar_simulacao(url_alvo)
