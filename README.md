# ClusterLens 

Aplicação web para classificação de texto via modelo hierárquico de clustering.

## Pipeline de Inferência

```
Texto → TF-IDF (500 features) → StandardScaler → PCA (50 componentes) → KNN → Cluster (0–3)
```

## Estrutura

```
app/
├── app.py                  # Backend Flask + API REST
├── Dockerfile              # Container de produção
├── docker-compose.yml      # Orquestração
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
# Build e execução
docker compose up --build

# Acesse: http://localhost:5000
```

### Sem Docker (dev)

```bash
pip install -r requirements.txt
python app.py
# Acesse: http://localhost:5000
```

## API REST

### `POST /api/predict`

**Body:**
```json
{ "text": "seu texto aqui" }
```

**Response:**
```json
{
  "cluster": "Cluster 1",
  "confidence": {
    "Cluster 0": 0.0,
    "Cluster 1": 100.0,
    "Cluster 2": 0.0,
    "Cluster 3": 0.0
  },
  "info": {
    "label": "Cluster 1",
    "icon": "▲",
    "color": "#4ECDC4",
    "description": "..."
  }
}
```

### `GET /api/health`

Retorna `{ "status": "ok", "models_loaded": true }`.
