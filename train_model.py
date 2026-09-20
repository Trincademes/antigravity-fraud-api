"""
Antigravity - Pipeline de Treinamento do Modelo de Detecção de Fraudes Enterprise
Autor: Arquiteto de Software & Cientista de Dados Sênior
Descrição: Gera dataset sintético de padrão bancário com múltiplos vetores de risco:
           idade da conta, tentativas falhas de login, reputação de conexão (VPN/Tor),
           novo favorecido, score bureau de crédito e biometria de dispositivo.
           Treina um classificador Random Forest calibrado e exporta o pipeline em 'fraud_model.pkl'.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Configuração de Logs Estruturados
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_model")

MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
RANDOM_SEED = 42
NUM_SAMPLES = 25_000
FRAUD_RATIO = 0.035  # 3.5% de fraudes

NUMERIC_FEATURES = [
    "valor",
    "hora_transacao",
    "tempo_desde_ultima_transacao",
    "distancia_localizacao_km",
    "score_dispositivo",
    "idade_conta_meses",
    "tentativas_falhas_24h",
    "score_credito_bureau"
]

CATEGORICAL_FEATURES = [
    "tipo_transacao",
    "tipo_conexao",
    "beneficiario_novo"
]


def gerar_dataset_sintetico(n_samples: int = NUM_SAMPLES, fraud_ratio: float = FRAUD_RATIO, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Gera dataset com telemetria bancária avançada:
    - ATO (Account Takeover), ataques por força bruta, desvio de fundos para laranjas/mulas
    - Anomalias de conexão (VPN/Tor), contas recém-criadas e bureaus de crédito.
    """
    np.random.seed(seed)
    n_frauds = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_frauds

    logger.info(f"Gerando {n_samples} transações bancárias ({n_legit} legítimas e {n_frauds} fraudulentas)...")

    # ==========================================
    # 1. PERFIL TRANSACIONAL LEGÍTIMO
    # ==========================================
    val_legit = np.random.lognormal(mean=4.3, sigma=0.9, size=n_legit).round(2)
    val_legit = np.clip(val_legit, 1.0, 18_000.0)

    prob_hora_legit = np.array([
        0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.08,
        0.08, 0.07, 0.07, 0.06, 0.06, 0.07, 0.07, 0.06, 0.04, 0.03, 0.02, 0.015
    ])
    prob_hora_legit = prob_hora_legit / prob_hora_legit.sum()
    hora_legit = np.random.choice(range(24), size=n_legit, p=prob_hora_legit)

    tempo_legit = np.random.exponential(scale=21_600, size=n_legit).round(1)
    tempo_legit = np.clip(tempo_legit, 60.0, 604_800.0)

    dist_legit = np.random.exponential(scale=6.0, size=n_legit).round(2)
    dist_legit = np.clip(dist_legit, 0.0, 120.0)

    score_legit = np.random.beta(a=9, b=1.5, size=n_legit).round(3)
    score_legit = np.clip(score_legit, 0.50, 1.0)

    # Idade da Conta (meses): contas maduras e fidelizadas (média 2.5 anos)
    idade_legit = np.random.gamma(shape=3.0, scale=10.0, size=n_legit).round(0)
    idade_legit = np.clip(idade_legit, 1.0, 180.0)

    # Tentativas Falhas de Senha/PIN nas últimas 24h: 96% zero, 3% uma falha, 1% duas
    tentativas_legit = np.random.choice([0, 1, 2], size=n_legit, p=[0.96, 0.03, 0.01])

    # Score de Crédito no Bureau (300 a 1000): média 740
    score_cred_legit = np.random.normal(loc=740, scale=110, size=n_legit).round(0)
    score_cred_legit = np.clip(score_cred_legit, 350.0, 990.0)

    # Beneficiário Novo (cadastrado há menos de 24h): raro (12%)
    benef_legit = np.random.choice(["NAO", "SIM"], size=n_legit, p=[0.88, 0.12])

    # Tipo de Conexão: residencial ou celular normal
    conexao_legit = np.random.choice(
        ["RESIDENCIAL", "MOVEL_4G_5G", "VPN_PROXY", "TOR"],
        size=n_legit,
        p=[0.60, 0.38, 0.019, 0.001]
    )

    tipo_legit = np.random.choice(
        ["PIX", "CARTAO_CREDITO", "BOLETO", "TED"],
        size=n_legit,
        p=[0.55, 0.30, 0.10, 0.05]
    )

    df_legit = pd.DataFrame({
        "valor": val_legit,
        "hora_transacao": hora_legit,
        "tempo_desde_ultima_transacao": tempo_legit,
        "distancia_localizacao_km": dist_legit,
        "score_dispositivo": score_legit,
        "idade_conta_meses": idade_legit,
        "tentativas_falhas_24h": tentativas_legit,
        "score_credito_bureau": score_cred_legit,
        "beneficiario_novo": benef_legit,
        "tipo_conexao": conexao_legit,
        "tipo_transacao": tipo_legit,
        "is_fraude": 0
    })

    # ==========================================
    # 2. PERFIL TRANSACIONAL FRAUDULENTO
    # ==========================================
    val_fraud = np.random.lognormal(mean=7.9, sigma=1.0, size=n_frauds).round(2)
    val_fraud = np.clip(val_fraud, 250.0, 50_000.0)

    # Madrugada / Horários de baixa vigilância
    prob_hora_fraud = np.array([
        0.13, 0.15, 0.15, 0.13, 0.10, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05
    ])
    prob_hora_fraud = prob_hora_fraud / prob_hora_fraud.sum()
    hora_fraud = np.random.choice(range(24), size=n_frauds, p=prob_hora_fraud)

    # Ataques em rajada (Burst) vs ataques espaçados
    is_burst = np.random.rand(n_frauds) < 0.65
    tempo_fraud = np.where(
        is_burst,
        np.random.exponential(scale=30.0, size=n_frauds).round(1),
        np.random.exponential(scale=2400.0, size=n_frauds).round(1)
    )
    tempo_fraud = np.clip(tempo_fraud, 1.0, 7200.0)

    # Distância Geográfica Anômala
    is_distant = np.random.rand(n_frauds) < 0.55
    dist_fraud = np.where(
        is_distant,
        np.random.uniform(low=200.0, high=3_500.0, size=n_frauds).round(2),
        np.random.exponential(scale=18.0, size=n_frauds).round(2)
    )

    # Score de Hardware / Dispositivo comprometido
    score_fraud = np.random.beta(a=1.2, b=6.0, size=n_frauds).round(3)
    score_fraud = np.clip(score_fraud, 0.01, 0.45)

    # Idade da Conta: Contas mulas recém-abertas (< 3 meses) ou sequestradas
    idade_fraud = np.random.choice(
        [1.0, 2.0, 3.0, 6.0, 12.0, 24.0],
        size=n_frauds,
        p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02]
    )

    # Tentativas Falhas prévias de autenticação (credential stuffing, force login)
    tentativas_fraud = np.random.choice(
        [0, 1, 2, 3, 4, 5],
        size=n_frauds,
        p=[0.15, 0.25, 0.30, 0.18, 0.08, 0.04]
    )

    # Score de Crédito: contas descartáveis têm score mais baixo
    score_cred_fraud = np.random.normal(loc=420, scale=130, size=n_frauds).round(0)
    score_cred_fraud = np.clip(score_cred_fraud, 150.0, 680.0)

    # Beneficiário Novo: 88% das fraudes transferem para chaves PIX/contas nunca vistas
    benef_fraud = np.random.choice(["NAO", "SIM"], size=n_frauds, p=[0.12, 0.88])

    # Conexão Anônima (VPN, Proxies, Tor)
    conexao_fraud = np.random.choice(
        ["RESIDENCIAL", "MOVEL_4G_5G", "VPN_PROXY", "TOR"],
        size=n_frauds,
        p=[0.05, 0.20, 0.50, 0.25]
    )

    tipo_fraud = np.random.choice(
        ["PIX", "CARTAO_CREDITO", "BOLETO", "TED"],
        size=n_frauds,
        p=[0.72, 0.24, 0.01, 0.03]
    )

    df_fraud = pd.DataFrame({
        "valor": val_fraud,
        "hora_transacao": hora_fraud,
        "tempo_desde_ultima_transacao": tempo_fraud,
        "distancia_localizacao_km": dist_fraud,
        "score_dispositivo": score_fraud,
        "idade_conta_meses": idade_fraud,
        "tentativas_falhas_24h": tentativas_fraud,
        "score_credito_bureau": score_cred_fraud,
        "beneficiario_novo": benef_fraud,
        "tipo_conexao": conexao_fraud,
        "tipo_transacao": tipo_fraud,
        "is_fraude": 1
    })

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def criar_pipeline_modelo() -> Pipeline:
    """
    Constrói a arquitetura do Scikit-Learn Pipeline com pré-processamento integrado.
    StandardScaler para features contínuas e OneHotEncoder para variáveis categóricas.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    classifier = RandomForestClassifier(
        n_estimators=140,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    return pipeline


def treinar_e_avaliar():
    """
    Treina o modelo nos dados sintéticos bancários, avalia métricas e persiste o pipeline.
    """
    logger.info("=== INICIANDO PIPELINE DE TREINAMENTO ANTIGRAVITY ENTERPRISE ===")

    df = gerar_dataset_sintetico()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_fraude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(f"Split de treino: {len(X_train)} amostras | Split de teste: {len(X_test)} amostras")

    pipeline = criar_pipeline_modelo()

    logger.info("Treinando modelo Random Forest com 11 features integradas...")
    pipeline.fit(X_train, y_train)

    logger.info("Avaliando modelo no conjunto de teste...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    logger.info(f"-> ROC-AUC Score: {roc_auc:.4f}")
    logger.info(f"\nRelatório de Classificação:\n{classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude'])}")
    logger.info(f"\nMatriz de Confusão:\n{confusion_matrix(y_test, y_pred)}")

    logger.info(f"Salvando artefato binário em '{MODEL_OUTPUT_PATH}'...")
    joblib.dump(pipeline, MODEL_OUTPUT_PATH, compress=3)
    logger.info("Artefato salvo com sucesso!")

    # Smoke Test
    logger.info("Executando Smoke Test com amostra de Invasão de Conta (ATO)...")
    modelo_carregado = joblib.load(MODEL_OUTPUT_PATH)
    amostra_teste = pd.DataFrame([{
        "valor": 12000.0,
        "hora_transacao": 3,
        "tempo_desde_ultima_transacao": 25.0,
        "distancia_localizacao_km": 150.0,
        "score_dispositivo": 0.05,
        "idade_conta_meses": 2.0,
        "tentativas_falhas_24h": 3,
        "score_credito_bureau": 380.0,
        "beneficiario_novo": "SIM",
        "tipo_conexao": "VPN_PROXY",
        "tipo_transacao": "PIX"
    }])
    prob_fraude = modelo_carregado.predict_proba(amostra_teste)[0][1]
    logger.info(f"Smoke Test: Probabilidade calculada para amostra ATO: {prob_fraude:.4f}")
    assert prob_fraude > 0.75, "O teste de sanidade falhou!"
    logger.info("=== TREINAMENTO CONCLUÍDO COM ÊXITO ===")


if __name__ == "__main__":
    treinar_e_avaliar()
