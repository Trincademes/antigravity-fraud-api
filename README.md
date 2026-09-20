# 🛡️ Antigravity - High Performance Fraud Detection API

API de missão crítica e alta performance desenvolvida em **Python** com **FastAPI** e **Scikit-Learn**, arquitetada para detecção preditiva de fraudes em transações financeiras em tempo real (sub-10ms de latência de inferência) e pronta para deploy contínuo no ambiente gratuito do **Render**.

---

## 📑 Sumário

- [Visão Geral e Arquitetura](#-visão-geral-e-arquitetura)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Matriz de Decisão e Regras de Negócio](#-matriz-de-decisão-e-regras-de-negócio)
- [Instalação e Execução Local](#-instalação-e-execução-local)
  - [1. Pré-requisitos](#1-pré-requisitos)
  - [2. Criação do Ambiente Virtual](#2-criação-do-ambiente-virtual)
  - [3. Instalação das Dependências](#3-instalação-das-dependências)
  - [4. Treinamento do Modelo](#4-treinamento-do-modelo)
  - [5. Execução do Servidor Uvicorn](#5-execução-do-servidor-uvicorn)
- [Guia de Endpoints e Testes Práticos](#-guia-de-endpoints-e-testes-práticos)
- [Deploy no Render (Passo a Passo Gratuito)](#-deploy-no-render-passo-a-passo-gratuito)
- [Boas Práticas de Engenharia e Produção](#-boas-práticas-de-engenharia-e-produção)

---

## 🏛️ Visão Geral e Arquitetura

O sistema **Antigravity** adota os princípios de **Clean Architecture** e **Defesa em Profundidade (Defense-in-Depth)**. O fluxo combina heurísticas de segurança determinísticas (regras duras de velocidade e geolocalização) com um pipeline de Machine Learning estocástico baseado em **Random Forest Classifier** com pesos balanceados (`balanced_subsample`).

```mermaid
flowchart TD
    Client([Cliente / Gateway Bancário]) -->|POST /v1/analisar-fraude| API[FastAPI Antigravity]
    API -->|1. Validação de Tipos e Limites| Pydantic[Pydantic v2 Schema]
    Pydantic -->|2. Checagem Determinística| HardRules{Viagem Impossível?}
    HardRules -->|Sim: Dist >= 1500km & Tempo < 10m| BlockHard[Decisão: BLOQUEADA (Hard Rule)]
    HardRules -->|Não| Pipeline[Scikit-Learn Pipeline]
    Pipeline -->|StandardScaler + OneHotEncoder| Preprocessing[Pré-processamento Vetorial]
    Preprocessing -->|Random Forest Classifier| Model[(fraud_model.pkl)]
    Model -->|Cálculo de Probabilidade| RiskEngine[Motor de Risco Antigravity]
    RiskEngine -->|Prob >= 0.80| Block[Decisão: BLOQUEADA]
    RiskEngine -->|0.30 <= Prob < 0.80| Review[Decisão: EM_ANALISE]
    RiskEngine -->|Prob < 0.30| Approve[Decisão: APROVADA]
    Block --> Response([Resposta JSON Estruturada])
    Review --> Response
    Approve --> Response
    BlockHard --> Response
    Response --> Client
```

---

## 📂 Estrutura do Projeto

```text
antigravity/
├── requirements.txt   # Especificação rígida de dependências de produção
├── train_model.py     # Gerador sintético realista, treinamento e serialização
├── main.py            # Servidor FastAPI com Lifespan, validação e endpoints
├── fraud_model.pkl    # Artefato binário do modelo treinado (gerado no build/treino)
└── README.md          # Manual técnico de arquitetura, execução e deploy
```

---

## ⚖️ Matriz de Decisão e Regras de Negócio

O motor avalia a probabilidade $P \in [0.0, 1.0]$ e classifica a transação:

| Faixa de Risco | Status de Decisão | Score de Risco | Ação Recomendada |
| :--- | :--- | :--- | :--- |
| **$P < 0.30$** | `APROVADA` | 0 a 29.9 | Autorização imediata da transação. |
| **$0.30 \le P < 0.80$** | `EM_ANALISE` | 30.0 a 79.9 | Desafio via biometria/2FA ou envio para mesa de analistas. |
| **$P \ge 0.80$** | `BLOQUEADA` | 80.0 a 100.0 | Bloqueio preventivo e emissão de alerta de segurança. |
| **Heurística Hard-Rule** | `BLOQUEADA` | 100.0 | Disparado em deslocamento $\ge 1500\text{ km}$ em intervalo $< 10\text{ min}$. |

---

## 💻 Instalação e Execução Local

### 1. Pré-requisitos
- Python 3.10, 3.11, 3.12 ou superior
- `pip` e `git` instalados

### 2. Criação do Ambiente Virtual
Recomenda-se o isolamento de dependências via `venv`:

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalação das Dependências
Instale as bibliotecas especificadas:

```bash
pip install -r requirements.txt
```

### 4. Treinamento do Modelo
Gere o dataset sintético de 20.000 transações balanceadas e treine o classificador:

```bash
python train_model.py
```
*Saída esperada:*
```text
[INFO] === INICIANDO PIPELINE DE TREINAMENTO ANTIGRAVITY ===
[INFO] Gerando 20000 transações (19400 legítimas e 600 fraudulentas)...
[INFO] Treinando modelo Random Forest com pré-processamento acoplado...
[INFO] -> ROC-AUC Score: 0.9880
[INFO] Salvando artefato binário em 'fraud_model.pkl'...
[INFO] Smoke Test: Probabilidade calculada para amostra de alto risco: 0.9412
[INFO] === TREINAMENTO CONCLUÍDO COM ÊXITO ===
```

### 5. Execução do Servidor Uvicorn
Inicie a API em modo de desenvolvimento:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Acesse a documentação interativa no navegador em: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧪 Guia de Endpoints e Testes Práticos

### 1. Health Check (`GET /`)
Utilizado pelo Render para verificação contínua de disponibilidade (Liveness & Readiness Probe).

```bash
curl -X GET "http://localhost:8000/"
```
*Resposta:*
```json
{
  "status": "healthy",
  "modelo_carregado": true,
  "versao": "1.0.0",
  "timestamp": "2026-09-19T21:30:00.000000Z",
  "ambiente": "production"
}
```

---

### 2. Transação Segura / Aprovada (`POST /v1/analisar-fraude`)
Transação habitual de baixo valor em horário comercial com dispositivo de confiança alta:

```bash
curl -X POST "http://localhost:8000/v1/analisar-fraude" \
     -H "Content-Type: application/json" \
     -d '{
       "id_transacao": "tx-legit-001",
       "id_usuario": "usr-88219",
       "valor": 85.50,
       "hora_transacao": 14,
       "tempo_desde_ultima_transacao": 21600.0,
       "distancia_localizacao_km": 3.2,
       "score_dispositivo": 0.98,
       "tipo_transacao": "PIX"
     }'
```
*Resposta Esperada:*
```json
{
  "id_transacao": "tx-legit-001",
  "id_usuario": "usr-88219",
  "status": "APROVADA",
  "probabilidade_fraude": 0.0215,
  "score_risco": 2.15,
  "motivo": "Transação segura (2.15/100). Operação dentro dos padrões habituais de consumo.",
  "latencia_ms": 3.42,
  "data_processamento": "2026-09-19T21:30:05.123456Z"
}
```

---

### 3. Transação Suspeita / Bloqueada (`POST /v1/analisar-fraude`)
Transação atípica de alto valor, na madrugada, de dispositivo desconhecido e alta distância:

```bash
curl -X POST "http://localhost:8000/v1/analisar-fraude" \
     -H "Content-Type: application/json" \
     -d '{
       "id_transacao": "tx-fraud-999",
       "id_usuario": "usr-12004",
       "valor": 12500.00,
       "hora_transacao": 3,
       "tempo_desde_ultima_transacao": 45.0,
       "distancia_localizacao_km": 1850.0,
       "score_dispositivo": 0.08,
       "tipo_transacao": "PIX"
     }'
```
*Resposta Esperada:*
```json
{
  "id_transacao": "tx-fraud-999",
  "id_usuario": "usr-12004",
  "status": "BLOQUEADA",
  "probabilidade_fraude": 0.9654,
  "score_risco": 96.54,
  "motivo": "Risco crítico detectado (96.54/100). Padrões severos de fraude identificados pelo classificador.",
  "latencia_ms": 4.15,
  "data_processamento": "2026-09-19T21:30:10.789101Z"
}
```

---

## 🚀 Deploy no Render (Passo a Passo Gratuito)

O Render permite hospedar aplicações FastAPI em seu plano **Free Web Service**.

### Passo 1: Enviar o Código ao GitHub
1. Crie um repositório no GitHub (ex: `antigravity-fraud-api`).
2. Adicione os arquivos do projeto e faça o push:
   ```bash
   git init
   git add .
   git commit -m "feat: initial commit Antigravity fraud engine"
   git branch -M main
   git remote add origin https://github.com/SEU_USUARIO/antigravity-fraud-api.git
   git push -u origin main
   ```

### Passo 2: Criar o Web Service no Render
1. Acesse o [Render Dashboard](https://dashboard.render.com/) e faça login.
2. Clique em **New +** no canto superior direito e selecione **Web Service**.
3. Selecione a opção **Build and deploy from a Git repository** e conecte seu repositório GitHub.

### Passo 3: Configurar os Parâmetros de Execução
Preencha o formulário de criação com as seguintes configurações:

| Campo | Valor Configurado | Explicação Técnica |
| :--- | :--- | :--- |
| **Name** | `antigravity-fraud-api` | Nome identificador do seu serviço. |
| **Region** | `Oregon (US West)` ou `Ohio` | Região com menor latência. |
| **Branch** | `main` | Branch de deploy contínuo. |
| **Runtime** | `Python 3` | Ambiente de execução oficial. |
| **Build Command** | `pip install -r requirements.txt && python train_model.py` | Instala pacotes e executa o treino para criar `fraud_model.pkl` automaticamente caso não esteja versionado no git! |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` | Inicia o servidor escutando na porta dinâmica injetada pelo Render. |
| **Instance Type** | `Free` | 512 MB RAM, 0.1 vCPU. |

### Passo 4: Configurar Health Check no Render
1. No menu lateral das configurações do serviço no Render, role até a seção **Advanced**.
2. No campo **Health Check Path**, insira:
   ```text
   /
   ```
3. O Render utilizará o endpoint `GET /` implementado no `main.py` para validar se a aplicação subiu com sucesso e se o modelo está pronto na memória antes de rotear tráfego público.

### Passo 5: Finalizar o Deploy
1. Clique no botão **Create Web Service**.
2. Acompanhe os logs da aba **Logs**:
   - O Render instalará os pacotes de `requirements.txt`.
   - Executará `train_model.py`, validando as métricas e gerando o arquivo `fraud_model.pkl`.
   - Executará o Uvicorn, iniciando a API.
3. Assim que o status mudar para **Live**, sua URL pública estará disponível (ex: `https://antigravity-fraud-api.onrender.com`).

> [!TIP]
> **Comportamento do Plano Gratuito do Render**:  
> No plano Free, o Render suspende instâncias que ficam sem tráfego por mais de 15 minutos (*spin-down*). Quando uma nova requisição chega, pode haver um *cold start* de cerca de 30 a 50 segundos. Para produção comercial sem suspensão, basta migrar para o plano *Starter* ou utilizar um monitor de ping externo (como UptimeRobot ou CronJob) apontando para o endpoint `/` a cada 10 minutos.

---

## 🔒 Boas Práticas de Engenharia e Produção

1. **Serialização Segura de Pipelines**: O Scikit-Learn `Pipeline` acopla o `ColumnTransformer` (StandardScaler + OneHotEncoder) junto ao estimador. Dessa forma, não há risco de *training-serving skew* (divergência entre transformações do treino e inferência).
2. **Lifespan Context**: O modelo é carregado apenas uma única vez na subida do processo worker, garantindo que as predições operem puramente em memória RAM com latência inferior a 10 milissegundos.
3. **Validação Estrita com Pydantic v2**: Prevenção de ataques de injeção de dados anômalos, estouro numérico e payloads corrompidos.
