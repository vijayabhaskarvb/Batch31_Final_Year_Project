import tensorflow as tf
print("TF version:", tf.__version__)

models = [
    ("models/brain_model.keras",    "models/brain_model.h5"),
    ("models/pneumonia_model.keras","models/pneumonia_model.h5"),
    ("models/dr_model.keras",       "models/dr_model.h5"),
]

for load_path, save_path in models:
    print(f"Loading {load_path}...")
    model = tf.keras.models.load_model(load_path)
    model.save(save_path)
    print(f"Saved as {save_path} ✓")

print("Done!")