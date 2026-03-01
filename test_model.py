"""
test_model.py — Interactive script to test the trained GRU model.

Usage:
    python test_model.py                        # interactive mode
    python test_model.py --input "some payload" # single prediction
    python test_model.py --file payloads.txt    # batch predictions from file
"""

import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration (must match train_model.py) ---
MAX_LEN = 300
MODEL_PATH = "gru_model.keras"
TOKENIZER_PATH = "tokenizer.pickle"
THRESHOLD = 0.5


def load_model_and_tokenizer():
    """Load the saved GRU model and tokenizer."""
    print("Loading model …")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  Model loaded from {MODEL_PATH}")

    print("Loading tokenizer …")
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    print(f"  Tokenizer loaded from {TOKENIZER_PATH}")

    return model, tokenizer


def predict(model, tokenizer, texts):
    """
    Predict whether each text is benign or malicious.

    Args:
        model: Keras model
        tokenizer: Fitted Keras Tokenizer
        texts: list of str

    Returns:
        list of dicts with keys: text, label, confidence
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0).flatten()

    results = []
    for text, prob in zip(texts, probabilities):
        label = "MALICIOUS" if prob > THRESHOLD else "BENIGN"
        confidence = prob if prob > THRESHOLD else 1 - prob
        results.append({
            "text": text,
            "label": label,
            "confidence": float(confidence),
            "raw_score": float(prob),
        })
    return results


def display_result(result, index=None):
    prefix = f"[{index}] " if index is not None else ""
    icon = "🔴" if result["label"] == "MALICIOUS" else "🟢"
    truncated = result["text"][:100] + ("…" if len(result["text"]) > 100 else "")
    print(
        f"{prefix}{icon} {result['label']}  "
        f"(confidence: {result['confidence']:.2%}, raw: {result['raw_score']:.4f})\n"
        f"    Input: {truncated}"
    )


def interactive_mode(model, tokenizer):
    """Run the model interactively — enter payloads one at a time."""
    print("\n" + "=" * 50)
    print("  GRU Web Attack Detector — Interactive Mode")
    print("  Type a payload and press Enter. Type 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        results = predict(model, tokenizer, [user_input])
        display_result(results[0])
        print()


def batch_mode(model, tokenizer, filepath):
    """Predict on every line of a text file."""
    print(f"\nReading payloads from {filepath} …")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"  {len(lines)} payloads loaded.\n")
    results = predict(model, tokenizer, lines)

    malicious_count = sum(1 for r in results if r["label"] == "MALICIOUS")
    benign_count = len(results) - malicious_count

    for i, r in enumerate(results, 1):
        display_result(r, index=i)

    print(f"\n--- Summary ---")
    print(f"  Total:     {len(results)}")
    print(f"  Malicious: {malicious_count}")
    print(f"  Benign:    {benign_count}")


def single_mode(model, tokenizer, text):
    """Predict on a single input string."""
    results = predict(model, tokenizer, [text])
    print()
    display_result(results[0])


# --- Example payloads for quick smoke-test ---
SAMPLE_PAYLOADS = [
    # Benign
    "/index.html",
    "GET /api/users?page=2 HTTP/1.1",
    "https://www.example.com/products/shoes",
    # XSS
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert(1)>",
    # SQL Injection
    "' OR 1=1 --",
    "'; DROP TABLE users; --",
    "1 UNION SELECT username,password FROM users",
    # Path traversal
    "../../etc/passwd",
    # Command injection
    "; cat /etc/shadow",
]


def demo_mode(model, tokenizer):
    """Run predictions on built-in sample payloads."""
    print("\n" + "=" * 50)
    print("  GRU Web Attack Detector — Demo Mode")
    print("=" * 50 + "\n")
    results = predict(model, tokenizer, SAMPLE_PAYLOADS)
    for i, r in enumerate(results, 1):
        display_result(r, index=i)

    malicious_count = sum(1 for r in results if r["label"] == "MALICIOUS")
    print(f"\n--- Demo Summary ---")
    print(f"  Total: {len(results)}  |  Malicious: {malicious_count}  |  Benign: {len(results) - malicious_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test the trained GRU web-attack detection model.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", "-i", type=str, help="Single payload string to classify.")
    group.add_argument("--file", "-f", type=str, help="Path to a text file with one payload per line.")
    group.add_argument("--demo", "-d", action="store_true", help="Run demo with built-in sample payloads.")

    args = parser.parse_args()
    model, tokenizer = load_model_and_tokenizer()

    if args.input:
        single_mode(model, tokenizer, args.input)
    elif args.file:
        batch_mode(model, tokenizer, args.file)
    elif args.demo:
        demo_mode(model, tokenizer)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
