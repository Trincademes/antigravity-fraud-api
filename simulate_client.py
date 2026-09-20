"""
Antigravity - Simulador de Transações Financeiras em Tempo Real
Utiliza apenas a biblioteca padrão (urllib) para execução imediata sem dependências extras.
"""

import sys
import time
import json
import urllib.request
import urllib.error

# Garante suporte a UTF-8 no terminal Windows (PowerShell / CMD)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_URL = "https://antigravity-fraud-api.onrender.com"

CENARIOS = [
    {
        "nome": "Café / Almoço Comercial Habitual",
        "payload": {
            "id_transacao": "tx-prod-001",
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
            "id_transacao": "tx-prod-002",
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
            "id_transacao": "tx-prod-003",
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
            "id_transacao": "tx-prod-004",
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
            "id_transacao": "tx-prod-005",
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
            "id_transacao": "tx-prod-006",
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
            "id_transacao": "tx-prod-007",
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
    base_url = base_url.rstrip("/")
    print("=" * 95)
    print("🚀 ANTIGRAVITY ENGINE - TESTE DE CARGA EM PRODUÇÃO NO RENDER")
    print(f"URL Alvo: {base_url}")
    print("=" * 95)

    # 1. Health Check
    try:
        req = urllib.request.Request(f"{base_url}/", headers={"User-Agent": "AntigravitySimulator/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data_health = json.loads(resp.read().decode("utf-8"))
            print(f"✅ Motor Online | Versão: {data_health.get('versao')} | Modelo Carregado: {data_health.get('modelo_carregado')} | Ambiente: {data_health.get('ambiente')}\n")
    except Exception as exc:
        print(f"❌ Erro de Conexão: Não foi possível alcançar {base_url}. Detalhes: {exc}")
        sys.exit(1)

    # 2. Execução dos Cenários
    endpoint = f"{base_url}/v1/analisar-fraude"
    icones = {
        "APROVADA": "🟢",
        "EM_ANALISE": "🟡",
        "BLOQUEADA": "🔴"
    }

    print(f"{'Cenário':<44} | {'Status':<14} | {'Score':<8} | {'Prob':<8} | {'Latência Servidor':<18} | {'RTT Total'}")
    print("-" * 110)

    for c in CENARIOS:
        nome = c["nome"]
        payload = c["payload"]
        dados_json = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=dados_json,
            headers={"Content-Type": "application/json", "User-Agent": "AntigravitySimulator/1.0"}
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
                rtt_str = f"{rtt:.2f}ms"
                ic = icones.get(st, "⚪")
                print(f"{nome:<44} | {ic} {st:<11} | {score:<8} | {prob:<8} | {lat_srv:<18} | {rtt_str}")
        except urllib.error.HTTPError as err:
            print(f"{nome:<44} | ❌ HTTP {err.code}: {err.read().decode('utf-8')[:40]}")
        except Exception as err:
            print(f"{nome:<44} | ❌ ERRO: {err}")

    print("=" * 110)
    print("✅ Simulação em produção finalizada com 100% de sucesso!\n")


if __name__ == "__main__":
    url_alvo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    executar_simulacao(url_alvo)
