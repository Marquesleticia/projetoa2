# SpamScan

Aplicação web para classificação de SMS em clusters via modelo hierárquico de clustering.

## Pipeline de Inferência 
Texto → TF-IDF (500 features) → StandardScaler → PCA (50 componentes) → KNN → Cluster (0–3)

## Clusters

| Cluster | Tag | Cor | Descrição |
|---|---|---|---|
| Cluster 0 ⚠ | SPAM AGRESSIVO | `#FF6B6B` | Alta pressão: prêmios, dinheiro imediato, chamadas urgentes |
| Cluster 1 ✉ | MENSAGEM LEGÍTIMA | `#4ECDC4` | SMS cotidiano e conversacional entre pessoas reais |
| Cluster 2 📢 | SPAM PROMOCIONAL | `#FFE66D` | Marketing, assinaturas, ringtones, concursos via SMS |
| Cluster 3 🔔 | NOTIFICAÇÃO / ALERTA | `#A78BFA` | SMS automáticos: banco, lembretes, confirmações de serviço |

# Estrutura do Projeto

```text
PROJETOA2/
├── app.py                  # Backend Flask + API REST
├── Dockerfile              # Container Docker
├── docker-compose.yml      # Orquestração dos serviços
├── requirements.txt        # Dependências Python
├── templates/
│   └── index.html          # Frontend
└── models/
    ├── tfidf_vectorizer.pkl
    ├── scaler.pkl
    ├── pca_model.pkl
    └── hclust_knn_model.pkl
```
## Como executar

### Com Docker (recomendado)

```bash
docker-compose up --build
# Acesse: http://localhost:5000
```

### Sem Docker (desenvolvimento)

```bash
source venv/bin/activate
pip install -r requirements.txt
python app.py
# Acesse: http://localhost:5001
```

## API REST

### `POST /api/predict`

**Body:**
```json
{ "text": "WINNER! You have won a £1000 cash prize. Call NOW!" }
```

**Response:**
```json
{
  "cluster": "Cluster 0",
  "confidence": {
    "Cluster 0": 95.0,
    "Cluster 1": 2.0,
    "Cluster 2": 3.0,
    "Cluster 3": 0.0
  },
  "info": {
    "label": "Cluster 0",
    "icon": "⚠",
    "color": "#FF6B6B",
    "tag": "SPAM AGRESSIVO",
    "description": "SMS com linguagem de alta pressão: prêmios, dinheiro imediato, números para ligar."
  }
}
```

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) — 4.457 mensagens rotuladas como spam ou ham.
