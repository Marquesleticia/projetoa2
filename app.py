import os
import warnings
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Carregar modelos na inicialização
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

tfidf    = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
pca      = joblib.load(os.path.join(MODEL_DIR, "pca_model.pkl"))
knn      = joblib.load(os.path.join(MODEL_DIR, "hclust_knn_model.pkl"))

CLUSTER_INFO = {
    "Cluster 0": {
        "label": "Cluster 0",
        "icon": "⚠",
        "color": "#FF6B6B",
        "tag": "SPAM AGRESSIVO",
        "description": "SMS com linguagem de alta pressão: prêmios, dinheiro imediato, números para ligar. Padrão de spam mais ostensivo — palavras como 'WINNER', 'CASH', 'URGENT', 'CLAIM NOW'.",
    },
    "Cluster 1": {
        "label": "Cluster 1",
        "icon": "✉",
        "color": "#4ECDC4",
        "tag": "MENSAGEM LEGÍTIMA",
        "description": "SMS cotidiano e conversacional: combinados, avisos pessoais, respostas curtas. Vocabulário informal típico de comunicação entre pessoas reais — 'ok', 'later', 'home', 'call you'.",
    },
    "Cluster 2": {
        "label": "Cluster 2",
        "icon": "📢",
        "color": "#FFE66D",
        "tag": "SPAM PROMOCIONAL",
        "description": "SMS de marketing e promoções: ofertas de serviços, assinaturas, ringtones, concursos via SMS. Linguagem persuasiva com instruções de resposta — 'reply STOP', 'txt YES', 'per week'.",
    },
    "Cluster 3": {
        "label": "Cluster 3",
        "icon": "🔔",
        "color": "#A78BFA",
        "tag": "NOTIFICAÇÃO / ALERTA",
        "description": "SMS automáticos e notificações de sistema: lembretes de conta, alertas de banco, confirmações de serviço. Tom formal e impessoal, diferente das mensagens humanas comuns.",
    },
}


def predict(text: str) -> dict:
    X      = tfidf.transform([text])
    X_sc   = scaler.transform(X.toarray())
    X_pca  = pca.transform(X_sc)
    label  = knn.predict(X_pca)[0]
    probas = knn.predict_proba(X_pca)[0]
    classes = knn.classes_

    confidence = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probas)}

    return {
        "cluster": label,
        "confidence": confidence,
        "info": CLUSTER_INFO.get(label, {}),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Texto não pode ser vazio."}), 400
    if len(text) < 5:
        return jsonify({"error": "Texto muito curto. Insira ao menos 5 caracteres."}), 400

    try:
        result = predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Erro na inferência: {str(e)}"}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models_loaded": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)