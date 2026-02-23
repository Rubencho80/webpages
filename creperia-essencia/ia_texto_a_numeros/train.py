import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un modelo texto->números sencillo")
    parser.add_argument("--dataset", required=True, help="Ruta al JSON con pares text/numbers")
    parser.add_argument("--model-out", default="modelo_texto_numeros.json", help="Ruta de salida del modelo")
    parser.add_argument("--epochs", type=int, default=2000, help="Número de épocas")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Tasa de aprendizaje")
    parser.add_argument("--hidden-size", type=int, default=32, help="Tamaño de la capa oculta")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--print-every", type=int, default=200, help="Frecuencia de impresión de pérdida")
    return parser.parse_args()


def load_dataset(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("El dataset debe ser una lista no vacía")

    texts = []
    targets = []
    expected_len = None

    for idx, item in enumerate(data):
        if "text" not in item or "numbers" not in item:
            raise ValueError(f"Elemento {idx} inválido: requiere keys 'text' y 'numbers'")
        text = item["text"]
        numbers = item["numbers"]

        if not isinstance(text, str):
            raise ValueError(f"Elemento {idx}: 'text' debe ser string")
        if not isinstance(numbers, list) or not numbers:
            raise ValueError(f"Elemento {idx}: 'numbers' debe ser lista no vacía")

        if expected_len is None:
            expected_len = len(numbers)
        elif len(numbers) != expected_len:
            raise ValueError(
                f"Elemento {idx}: longitud de 'numbers' distinta ({len(numbers)} != {expected_len})"
            )

        texts.append(text)
        targets.append([float(n) for n in numbers])

    return texts, targets


def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    return {ch: i + 1 for i, ch in enumerate(chars)}


def encode_text(text, vocab, max_len):
    values = [0.0] * max_len
    for i, ch in enumerate(text[:max_len]):
        values[i] = float(vocab.get(ch, 0))
    vocab_size = max(len(vocab), 1)
    return [v / vocab_size for v in values]


def relu(x):
    return x if x > 0 else 0.0


def relu_grad(x):
    return 1.0 if x > 0 else 0.0


def init_matrix(rows, cols, rng, std=0.2):
    return [[rng.uniform(-std, std) for _ in range(cols)] for _ in range(rows)]


def init_vector(size):
    return [0.0] * size


def dot(vec, mat):
    cols = len(mat[0])
    out = [0.0] * cols
    for j in range(cols):
        s = 0.0
        for i, v in enumerate(vec):
            s += v * mat[i][j]
        out[j] = s
    return out


def train_model(x_data, y_data, hidden_size, learning_rate, epochs, print_every, rng):
    n_samples = len(x_data)
    input_size = len(x_data[0])
    output_size = len(y_data[0])

    w1 = init_matrix(input_size, hidden_size, rng)
    b1 = init_vector(hidden_size)
    w2 = init_matrix(hidden_size, output_size, rng)
    b2 = init_vector(output_size)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for x, y_true in zip(x_data, y_data):
            z1 = [a + b for a, b in zip(dot(x, w1), b1)]
            a1 = [relu(v) for v in z1]
            y_pred = [a + b for a, b in zip(dot(a1, w2), b2)]

            sample_loss = sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / output_size
            total_loss += sample_loss

            grad_y = [2.0 * (p - t) / output_size for p, t in zip(y_pred, y_true)]

            grad_w2 = [[0.0] * output_size for _ in range(hidden_size)]
            grad_b2 = grad_y[:]

            for h in range(hidden_size):
                for o in range(output_size):
                    grad_w2[h][o] = a1[h] * grad_y[o]

            grad_a1 = [0.0] * hidden_size
            for h in range(hidden_size):
                grad_a1[h] = sum(grad_y[o] * w2[h][o] for o in range(output_size))

            grad_z1 = [grad_a1[h] * relu_grad(z1[h]) for h in range(hidden_size)]

            grad_w1 = [[0.0] * hidden_size for _ in range(input_size)]
            grad_b1 = grad_z1[:]

            for i in range(input_size):
                for h in range(hidden_size):
                    grad_w1[i][h] = x[i] * grad_z1[h]

            for h in range(hidden_size):
                for o in range(output_size):
                    w2[h][o] -= learning_rate * grad_w2[h][o]
            for o in range(output_size):
                b2[o] -= learning_rate * grad_b2[o]

            for i in range(input_size):
                for h in range(hidden_size):
                    w1[i][h] -= learning_rate * grad_w1[i][h]
            for h in range(hidden_size):
                b1[h] -= learning_rate * grad_b1[h]

        avg_loss = total_loss / n_samples
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs} - loss: {avg_loss:.6f}")

    return w1, b1, w2, b2


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    dataset_path = Path(args.dataset)
    texts, y = load_dataset(dataset_path)

    vocab = build_vocab(texts)
    max_len = max(len(t) for t in texts)
    x = [encode_text(t, vocab, max_len) for t in texts]

    w1, b1, w2, b2 = train_model(
        x,
        y,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        print_every=args.print_every,
        rng=rng,
    )

    model = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "max_len": max_len,
        "output_size": len(y[0]),
        "vocab": vocab,
    }

    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")

    print(f"Modelo guardado en: {output_path}")


if __name__ == "__main__":
    main()
