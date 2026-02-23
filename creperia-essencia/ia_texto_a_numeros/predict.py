import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Predice secuencias numéricas desde texto")
    parser.add_argument("--model", required=True, help="Ruta del modelo .json")
    parser.add_argument("--text", required=True, help="Texto de entrada")
    return parser.parse_args()


def relu(x):
    return x if x > 0 else 0.0


def dot(vec, mat):
    cols = len(mat[0])
    out = [0.0] * cols
    for j in range(cols):
        s = 0.0
        for i, v in enumerate(vec):
            s += v * mat[i][j]
        out[j] = s
    return out


def encode_text(text, vocab, max_len):
    encoded = [0.0] * max_len
    for i, ch in enumerate(text[:max_len]):
        encoded[i] = float(vocab.get(ch, 0))
    vocab_size = max(len(vocab), 1)
    return [x / vocab_size for x in encoded]


def main():
    args = parse_args()
    model_path = Path(args.model)
    model = json.loads(model_path.read_text(encoding="utf-8"))

    w1 = model["w1"]
    b1 = model["b1"]
    w2 = model["w2"]
    b2 = model["b2"]
    max_len = model["max_len"]
    vocab = model["vocab"]

    x = encode_text(args.text, vocab, max_len)
    z1 = [a + b for a, b in zip(dot(x, w1), b1)]
    a1 = [relu(v) for v in z1]
    y_pred = [a + b for a, b in zip(dot(a1, w2), b2)]

    result = [round(v, 4) for v in y_pred]

    print(f"Texto: {args.text}")
    print(f"Predicción: {result}")


if __name__ == "__main__":
    main()
