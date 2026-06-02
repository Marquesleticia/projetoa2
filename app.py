import os
import warnings

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Carregar modelos na inicializacao
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
pca = joblib.load(os.path.join(MODEL_DIR, "pca_model.pkl"))
knn = joblib.load(os.path.join(MODEL_DIR, "hclust_knn_model.pkl"))

# Ajuste do KNN usado como ponte de inferencia do HClust.
# k=1 deixava as probabilidades sempre em 0% ou 100%.
# k=11 + pesos por distancia gera distribuicoes mais uteis de confianca.
knn.set_params(n_neighbors=11, weights="distance")

CLUSTER_INFO = {
    "Cluster 0": {
        "label": "Cluster 0",
        "class_label": "SPAM",
        "icon": "⚠",
        "color": "#FF6B6B",
        "tag": "SPAM AGRESSIVO",
        "description": "SMS com linguagem de alta pressao: premios, dinheiro imediato e numeros para ligar. Padrao de spam mais ostensivo, com termos como WINNER, CASH, URGENT e CLAIM NOW.",
    },
    "Cluster 1": {
        "label": "Cluster 1",
        "class_label": "HAM",
        "icon": "✉",
        "color": "#4ECDC4",
        "tag": "MENSAGEM LEGITIMA",
        "description": "SMS cotidiano e conversacional: combinados, avisos pessoais e respostas curtas. Vocabulario informal tipico de comunicacao entre pessoas reais.",
    },
    "Cluster 2": {
        "label": "Cluster 2",
        "class_label": "SPAM",
        "icon": "📢",
        "color": "#FFE66D",
        "tag": "SPAM PROMOCIONAL",
        "description": "SMS de marketing e promocoes: ofertas de servicos, assinaturas, concursos via SMS e linguagem persuasiva com instrucoes de resposta.",
    },
    "Cluster 3": {
        "label": "Cluster 3",
        "class_label": "HAM",
        "icon": "🔔",
        "color": "#A78BFA",
        "tag": "NOTIFICACAO / ALERTA",
        "description": "SMS automaticos e notificacoes de sistema: lembretes de conta, alertas de banco ou confirmacoes de servico. Em geral, possui tom mais formal e impessoal.",
    },
}

LABEL_INFO = {
    "SPAM": {
        "text": "SPAM",
        "color": "#FF6B6B",
        "description": "Mensagem com padrao mais proximo de spam ou comunicacao promocional.",
    },
    "HAM": {
        "text": "HAM",
        "color": "#4ECDC4",
        "description": "Mensagem com padrao mais proximo de comunicacao legitima ou nao promocional.",
    },
}


def normalize_cluster_label(label) -> str:
    """Garante que o nome do cluster venha no formato usado pelo frontend."""
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    if isinstance(label, (np.integer, int)):
        return f"Cluster {int(label)}"
    label = str(label)
    if label.lower().startswith("cluster"):
        return label
    if label.isdigit():
        return f"Cluster {label}"
    return label


def extract_top_tokens(tfidf_row, top_n: int = 5) -> list[dict]:
    """Retorna os tokens com maior peso TF-IDF na mensagem analisada."""
    if tfidf_row.nnz == 0:
        return []

    feature_names = tfidf.get_feature_names_out()
    row = tfidf_row.tocsr()[0]

    order = np.argsort(row.data)[::-1][:top_n]
    tokens = []
    for pos in order:
        token_index = row.indices[pos]
        weight = row.data[pos]
        tokens.append({
            "token": str(feature_names[token_index]),
            "weight": round(float(weight), 4),
        })
    return tokens


def predict(text: str) -> dict:
    X_tfidf = tfidf.transform([text])
    X_scaled = scaler.transform(X_tfidf.toarray())
    X_pca = pca.transform(X_scaled)

    raw_label = knn.predict(X_pca)[0]
    cluster = normalize_cluster_label(raw_label)

    probas = knn.predict_proba(X_pca)[0]
    classes = [normalize_cluster_label(cls) for cls in knn.classes_]
    confidence = {
        cls: round(float(prob) * 100, 1)
        for cls, prob in zip(classes, probas)
    }

    distances, _ = knn.kneighbors(X_pca, n_neighbors=1, return_distance=True)
    nearest_distance = round(float(distances[0][0]), 4)

    info = CLUSTER_INFO.get(cluster, {})
    label = info.get("class_label", "SPAM" if cluster in {"Cluster 0", "Cluster 2"} else "HAM")

    return {
        "label": label,
        "label_info": LABEL_INFO.get(label, {}),
        "cluster": cluster,
        "confidence": confidence,
        "distance": nearest_distance,
        "top_tokens": extract_top_tokens(X_tfidf, top_n=5),
        "info": info,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Texto nao pode ser vazio."}), 400
    if len(text) < 5:
        return jsonify({"error": "Texto muito curto. Insira ao menos 5 caracteres."}), 400

    try:
        result = predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Erro na inferencia: {str(e)}"}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "knn_n_neighbors": knn.get_params().get("n_neighbors"),
        "knn_weights": knn.get_params().get("weights"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
