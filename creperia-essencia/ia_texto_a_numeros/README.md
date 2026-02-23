# Proyecto IA: texto ➜ secuencia numérica

Este proyecto entrena un modelo sencillo para convertir una secuencia de texto en una secuencia de números.

## ¿Qué incluye?

- `train.py`: entrena un modelo neuronal básico (implementado en Python puro, sin librerías externas).
- `predict.py`: carga el modelo entrenado y predice números para un texto.
- `dataset_ejemplo.json`: dataset de ejemplo para que puedas arrancar rápido.

## 1) Preparar tu dataset

El dataset debe ser un JSON con esta estructura:

```json
[
  {"text": "abc", "numbers": [1, 2, 3]},
  {"text": "cab", "numbers": [3, 1, 2]}
]
```

Reglas:
- Todos los elementos deben tener `text` (string) y `numbers` (lista de números).
- Todas las listas `numbers` deben tener **la misma longitud**.

## 2) Entrenar

```bash
python train.py \
  --dataset dataset_ejemplo.json \
  --model-out modelo_texto_numeros.json \
  --epochs 2000 \
  --learning-rate 0.03 \
  --hidden-size 32
```

## 3) Predecir

```bash
python predict.py \
  --model modelo_texto_numeros.json \
  --text "abc"
```

Salida esperada (ejemplo):

```text
Texto: abc
Predicción: [1.01, 1.98, 3.02]
```

## Notas

- Es un modelo simple y didáctico, pensado para experimentar.
- Si quieres más precisión, añade más ejemplos de entrenamiento y más épocas.
- Si tu texto contiene caracteres nuevos no vistos en entrenamiento, el modelo los ignora.
