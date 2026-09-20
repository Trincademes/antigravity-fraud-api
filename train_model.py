"""
Antigravity - Pipeline de Treinamento do Modelo de Detecção de Fraudes
Autor: Arquiteto de Software & Cientista de Dados Sênior
Descrição: Gera dataset sintético com padrões verossímeis de transações bancárias,
           treina um classificador Random Forest calibrado e exporta o pipeline
           completo (pré-processamento + modelo) em 'fraud_model.pkl'.
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

# Constantes do Modelo
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
RANDOM_SEED = 42
NUM_SAMPLES = 20_000
FRAUD_RATIO = 0.03  # 3% de fraudes (desbalanceamento realista)

NUMERIC_FEATURES = [
    "valor",
    "hora_transacao",
    "tempo_desde_ultima_transacao",
    "distancia_localizacao_km",
    "score_dispositivo"
]

CATEGORICAL_FEATURES = [
    "tipo_transacao"
]


def gerar_dataset_sintetico(n_samples: int = NUM_SAMPLES, fraud_ratio: float = FRAUD_RATIO, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Gera dataset sintético simulando padrões comportamentais reais de transações financeiras.
    - Transações Legítimas: valores moderados, horários comerciais, dispositivos confiáveis, distâncias curtas.
    - Transações Fraudulentas: picos em madrugadas, valores atípicos, alta velocidade/frequência, dispositivos suspeitos.
    """
    np.random.seed(seed)
    n_frauds = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_frauds

    logger.info(f"Gerando {n_samples} transações ({n_legit} legítimas e {n_frauds} fraudulentas)...")

    # --- 1. Dados Legítimos ---
    # Valores: distribuição log-normal (maioria entre R$ 20 e R$ 400, raros acima de R$ 2.000)
    val_legit = np.random.lognormal(mean=4.2, sigma=0.9, size=n_legit).round(2)
    val_legit = np.clip(val_legit, 1.0, 15_000.0)

    # Horas: pico entre 08:00 e 22:00
    prob_hora_legit = np.array([
        0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.08,
        0.08, 0.07, 0.07, 0.06, 0.06, 0.07, 0.07, 0.06, 0.04, 0.03, 0.02, 0.015
    ])
    prob_hora_legit = prob_hora_legit / prob_hora_legit.sum()
    hora_legit = np.random.choice(range(24), size=n_legit, p=prob_hora_legit)

    # Tempo desde a última transação (segundos): intervalo normal (ex: minutos a dias)
    tempo_legit = np.random.exponential(scale=18_000, size=n_legit).round(1)
    tempo_legit = np.clip(tempo_legit, 120.0, 604_800.0)

    # Distância geográfica do ponto habitual (km): local habitual (0 a 30 km)
    dist_legit = np.random.exponential(scale=8.0, size=n_legit).round(2)
    dist_legit = np.clip(dist_legit, 0.0, 150.0)

    # Score de reputação do dispositivo: alto (0.75 a 1.0)
    score_legit = np.random.beta(a=9, b=1.5, size=n_legit).round(3)
    score_legit = np.clip(score_legit, 0.50, 1.0)

    # Canal / Tipo de Transação
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
        "tipo_transacao": tipo_legit,
        "is_fraude": 0
    })

    # --- 2. Dados Fraudulentos (Modelagem com Múltiplas Tipologias de Fraude) ---
    # Tipologias representadas:
    # 1. Invasão de Conta (ATO): valor alto, score baixo, madrugada.
    # 2. Ataque de Velocidade (Burst): tempo muito curto entre transações, dispositivo suspeito.
    # 3. Anomalia Geográfica: distância muito elevada da residência.
    # 4. Saque Noturno Imediato: PIX de madrugada com valores elevados.
    val_fraud = np.random.lognormal(mean=7.8, sigma=1.1, size=n_frauds).round(2)
    val_fraud = np.clip(val_fraud, 300.0, 50_000.0)

    # Horas: picos expressivos na madrugada (00:00 às 05:00)
    prob_hora_fraud = np.array([
        0.12, 0.14, 0.14, 0.12, 0.10, 0.06, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05
    ])
    prob_hora_fraud = prob_hora_fraud / prob_hora_fraud.sum()
    hora_fraud = np.random.choice(range(24), size=n_frauds, p=prob_hora_fraud)

    # Tempo desde a última transação: 60% são rajadas (< 90s), 40% são intervalos variáveis
    is_burst = np.random.rand(n_frauds) < 0.60
    tempo_fraud = np.where(
        is_burst,
        np.random.exponential(scale=35.0, size=n_frauds).round(1),
        np.random.exponential(scale=1800.0, size=n_frauds).round(1)
    )
    tempo_fraud = np.clip(tempo_fraud, 1.0, 7200.0)

    # Distância geográfica: mistura de anomalias locais (dispositivo adulterado) e remotas
    is_distant = np.random.rand(n_frauds) < 0.50
    dist_fraud = np.where(
        is_distant,
        np.random.uniform(low=250.0, high=3_500.0, size=n_frauds).round(2),
        np.random.exponential(scale=20.0, size=n_frauds).round(2)
    )

    # Score do dispositivo: predominantemente baixo (0.01 a 0.40)
    score_fraud = np.random.beta(a=1.2, b=6.0, size=n_frauds).round(3)
    score_fraud = np.clip(score_fraud, 0.01, 0.45)

    # Canal / Tipo: PIX e Cartão sem presença física dominam os ataques
    tipo_fraud = np.random.choice(
        ["PIX", "CARTAO_CREDITO", "BOLETO", "TED"],
        size=n_frauds,
        p=[0.70, 0.25, 0.01, 0.04]
    )

    df_fraud = pd.DataFrame({
        "valor": val_fraud,
        "hora_transacao": hora_fraud,
        "tempo_desde_ultima_transacao": tempo_fraud,
        "distancia_localizacao_km": dist_fraud,
        "score_dispositivo": score_fraud,
        "tipo_transacao": tipo_fraud,
        "is_fraude": 1
    })

    # Unificação e embaralhamento
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def criar_pipeline_modelo() -> Pipeline:
    """
    Constrói a arquitetura do Scikit-Learn Pipeline com pré-processamento integrado.
    Garante idempotência: os dados de entrada na API não requerem transformações manuais.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=3,
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
    Função principal: gera dados, treina o modelo, avalia métricas essenciais e persiste o artefato.
    """
    logger.info("=== INICIANDO PIPELINE DE TREINAMENTO ANTIGRAVITY ===")

    df = gerar_dataset_sintetico()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_fraude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(f"Split de treino: {len(X_train)} amostras | Split de teste: {len(X_test)} amostras")

    pipeline = criar_pipeline_modelo()

    logger.info("Treinando modelo Random Forest com pré-processamento acoplado...")
    pipeline.fit(X_train, y_train)

    logger.info("Avaliando modelo no conjunto de teste...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    logger.info(f"-> ROC-AUC Score: {roc_auc:.4f}")
    logger.info(f"\nRelatório de Classificação:\n{classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude'])}")
    logger.info(f"\nMatriz de Confusão:\n{confusion_matrix(y_test, y_pred)}")

    # Salva o binário do modelo com compressão otimizada
    logger.info(f"Salvando artefato binário em '{MODEL_OUTPUT_PATH}'...")
    joblib.dump(pipeline, MODEL_OUTPUT_PATH, compress=3)
    logger.info("Artefato salvo com sucesso!")

    # Teste de Sanidade (Smoke Test com predição unitária)
    logger.info("Executando Smoke Test com predição unitária...")
    modelo_carregado = joblib.load(MODEL_OUTPUT_PATH)
    amostra_teste = pd.DataFrame([{
        "valor": 7500.0,
        "hora_transacao": 3,
        "tempo_desde_ultima_transacao": 15.0,
        "distancia_localizacao_km": 1200.0,
        "score_dispositivo": 0.12,
        "tipo_transacao": "PIX"
    }])
    prob_fraude = modelo_carregado.predict_proba(amostra_teste)[0][1]
    logger.info(f"Smoke Test: Probabilidade calculada para amostra de alto risco: {prob_fraude:.4f}")
    assert prob_fraude > 0.5, "O teste de sanidade falhou: a amostra de alto risco não foi identificada!"
    logger.info("=== TREINAMENTO CONCLUÍDO COM ÊXITO ===")


if __name__ == "__main__":
    treinar_e_avaliar()
