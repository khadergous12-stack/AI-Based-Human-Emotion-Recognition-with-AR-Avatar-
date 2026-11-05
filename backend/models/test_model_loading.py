import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from backend.models.model_architecture import build_emotion_cnn

WEIGHTS_PATH = "backend/models/emotion_custom_cnn_weights.npz"

# Build the CNN architecture
model = build_emotion_cnn()

print("Local CNN architecture created!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Total parameters:", model.count_params())

# Load exported NumPy weights
with np.load(WEIGHTS_PATH, allow_pickle=False) as data:

    for index, layer in enumerate(model.layers):
        expected_weights = layer.get_weights()

        if not expected_weights:
            continue

        transferred_weights = [
            data[f"layer_{index}_weight_{weight_index}"]
            for weight_index in range(len(expected_weights))
        ]

        # Verify every weight shape before loading
        for expected, transferred in zip(
            expected_weights, transferred_weights
        ):
            if expected.shape != transferred.shape:
                raise ValueError(
                    f"Weight mismatch in layer {layer.name}: "
                    f"expected {expected.shape}, "
                    f"received {transferred.shape}"
                )

        layer.set_weights(transferred_weights)

print("\nAll trained weights transferred successfully!")

# Test prediction
sample_image = np.random.random(
    (1, 48, 48, 1)
).astype(np.float32)

prediction = model.predict(sample_image, verbose=0)

print("\nPrediction test successful!")
print("Prediction shape:", prediction.shape)
print("Predicted class index:", int(np.argmax(prediction)))
print("Prediction probabilities:", prediction[0])
print("Probability sum:", float(np.sum(prediction[0])))
