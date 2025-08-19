from __future__ import annotations
import argparse
from pathlib import Path

import gradio as gr
from joblib import load
from preprocess import preprocess_text_for_inference


def parse_args():
    p = argparse.ArgumentParser(description="Launch UI for sentiment analysis")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def build_interface(available_models: dict[str, tuple]):
    """
    available_models: mapping from display name -> (vectorizer, model)
    """
    model_names = list(available_models.keys())

    def predict_fn(model_name: str, text: str):
        if not text or not text.strip():
            return "—", 0.0
        if model_name not in available_models:
            return "Model not loaded", 0.0
        vec, model = available_models[model_name]
        cleaned = preprocess_text_for_inference(text)
        X = vec.transform([cleaned])
        try:
            proba_pos = float(model.predict_proba(X)[0][1])
        except Exception:
            # Fallback if predict_proba is unavailable
            try:
                from scipy.special import expit
                score = float(model.decision_function(X)[0])
                proba_pos = float(expit(score))
            except Exception:
                proba_pos = float(model.predict(X)[0])
        label = "Positive ✅" if proba_pos >= 0.5 else "Negative ❌"
        return label, round(proba_pos, 4)

    with gr.Blocks(title="Sentiment Analysis") as demo:
        gr.Markdown("# Sentiment Analysis\nSelect a model and enter a review to predict sentiment.")
        model_dd = gr.Dropdown(choices=model_names, value=model_names[0], label="Model")
        inp = gr.Textbox(label="Review text", placeholder="Type or paste a review…", lines=6)
        btn = gr.Button("Predict")
        out_label = gr.Textbox(label="Sentiment")
        out_prob = gr.Number(label="P(Positive)")
        btn.click(predict_fn, inputs=[model_dd, inp], outputs=[out_label, out_prob])
    return demo


def main():
    args = parse_args()
    # Detect available models
    expected = {
        "Naive Bayes": (Path("models/tfidf_nb.joblib"), Path("models/nb_model.joblib")),
        "Logistic Regression": (Path("models/tfidf_logreg.joblib"), Path("models/logreg_model.joblib")),
    }
    available = {}
    missing = []
    for name, (vec_p, model_p) in expected.items():
        if vec_p.exists() and model_p.exists():
            available[name] = (load(str(vec_p)), load(str(model_p)))
        else:
            missing.append((name, vec_p, model_p))

    if not available:
        lines = [
            "No trained models found. Train models first:",
            "  python train.py nb --data_dir data",
            "  python train.py logreg --data_dir data",
            "Expected artifacts (missing):",
        ]
        for name, vec_p, model_p in missing:
            lines.append(f"  {name}: {vec_p} | {model_p}")
        raise SystemExit("\n".join(lines))

    demo = build_interface(available)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
