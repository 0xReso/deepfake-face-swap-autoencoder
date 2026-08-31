"""
Stage 2/4 - Model Training

Trains a shared encoder with two identity-specific decoders using a
VGG16 perceptual loss and mixed-precision on GPU.

Input:  face crops produced by 01_preprocessing.py
Output: encoder / decoder_a / decoder_b .weights.h5

Kaggle: https://www.kaggle.com/code/ayushragh/deepfakes-model-training
Adapted from: https://www.kaggle.com/code/sergiovirahonda/deepfakes-model-training
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split

# ============================================================
# MIXED PRECISION
# ============================================================
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

FACE_SIZE = 64
LATENT_DIM = 1024
EPOCHS = 300
BATCH_SIZE = 32

# ============================================================
# PERCEPTUAL LOSS — defined BEFORE any model building/compiling
# so it's available everywhere
# ============================================================
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(FACE_SIZE, FACE_SIZE, 3))
vgg.trainable = False
for layer in vgg.layers:
    layer.trainable = False
perceptual_model = keras.Model(vgg.input, vgg.get_layer('block2_conv2').output)
perceptual_model.trainable = False

def perceptual_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    feat_true = tf.cast(perceptual_model(y_true, training=False), tf.float32)
    feat_pred = tf.cast(perceptual_model(y_pred, training=False), tf.float32)
    pixel_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    feat_loss = tf.reduce_mean(tf.abs(feat_true - feat_pred))
    return pixel_loss + 0.5 * feat_loss

# ============================================================
# DATASET CREATION
# ============================================================
def create_dataset(path):
    images = []
    for dirname, _, filenames in os.walk(path):
        for filename in sorted(filenames):
            filepath = os.path.join(dirname, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype('float32') / 255.0
            images.append(image)
    return np.array(images)

faces_1 = create_dataset('/kaggle/input/notebooks/ayushragh/deepfakes-preprocessing/results_1/')
faces_2 = create_dataset('/kaggle/input/notebooks/ayushragh/deepfakes-preprocessing/results_2/')

print(f"Person A samples: {len(faces_1)}")
print(f"Person B samples: {len(faces_2)}")

# ============================================================
# DATASET SPLITTING
# ============================================================
X_train_a, X_test_a = train_test_split(faces_1, test_size=0.15, random_state=42)
X_train_b, X_test_b = train_test_split(faces_2, test_size=0.15, random_state=42)

print(f"Train A: {X_train_a.shape}, Test A: {X_test_a.shape}")
print(f"Train B: {X_train_b.shape}, Test B: {X_test_b.shape}")

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
def build_encoder():
    inp = layers.Input(shape=(FACE_SIZE, FACE_SIZE, 3))
    x = layers.Conv2D(128, 5, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(256, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(512, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(1024, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(LATENT_DIM)(x)
    x = layers.Dense(4 * 4 * 512)(x)
    x = layers.Reshape((4, 4, 512))(x)
    return keras.Model(inp, x, name="encoder")


def build_decoder(name="decoder"):
    inp = layers.Input(shape=(4, 4, 512))
    x = layers.Conv2DTranspose(512, 5, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2DTranspose(256, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2DTranspose(128, 5, strides=2, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2DTranspose(3, 5, strides=2, padding='same')(x)
    x = layers.Activation('sigmoid', dtype='float32')(x)
    return keras.Model(inp, x, name=name)


encoder   = build_encoder()
decoder_a = build_decoder(name="decoder_a")
decoder_b = build_decoder(name="decoder_b")

encoder.summary()
decoder_a.summary()

auto_input = layers.Input(shape=(FACE_SIZE, FACE_SIZE, 3))
latent = encoder(auto_input)

autoencoder_a = keras.Model(auto_input, decoder_a(latent), name="autoencoder_a")
autoencoder_b = keras.Model(auto_input, decoder_b(latent), name="autoencoder_b")

autoencoder_a.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.999),
    loss=perceptual_loss
)
autoencoder_b.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.999),
    loss=perceptual_loss
)

# ============================================================
# ALTERNATING TRAINING
# ============================================================
print("\n=== Starting alternating training ===\n")

best_val_a = float('inf')
best_val_b = float('inf')
patience_counter = 0
MAX_PATIENCE = 40
INITIAL_LR = 5e-5
current_lr = INITIAL_LR

history_a_loss, history_a_val = [], []
history_b_loss, history_b_val = [], []

for epoch in range(EPOCHS):
    hist_a = autoencoder_a.fit(
        X_train_a, X_train_a,
        epochs=1, batch_size=BATCH_SIZE, shuffle=True,
        validation_data=(X_test_a, X_test_a), verbose=0
    )
    hist_b = autoencoder_b.fit(
        X_train_b, X_train_b,
        epochs=1, batch_size=BATCH_SIZE, shuffle=True,
        validation_data=(X_test_b, X_test_b), verbose=0
    )

    tl_a = hist_a.history['loss'][0]
    vl_a = hist_a.history['val_loss'][0]
    tl_b = hist_b.history['loss'][0]
    vl_b = hist_b.history['val_loss'][0]

    history_a_loss.append(tl_a)
    history_a_val.append(vl_a)
    history_b_loss.append(tl_b)
    history_b_val.append(vl_b)

    improved = False
    if vl_a < best_val_a:
        best_val_a = vl_a
        autoencoder_a.save_weights('autoencoder_a.weights.h5')
        improved = True
    if vl_b < best_val_b:
        best_val_b = vl_b
        autoencoder_b.save_weights('autoencoder_b.weights.h5')
        improved = True

    if improved:
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter > 0 and patience_counter % 15 == 0:
        current_lr *= 0.5
        tf.keras.backend.set_value(autoencoder_a.optimizer.learning_rate, current_lr)
        tf.keras.backend.set_value(autoencoder_b.optimizer.learning_rate, current_lr)
        print(f"  >> Reduced LR to {current_lr:.2e}")

    if epoch % 10 == 0 or improved:
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"A: train={tl_a:.5f} val={vl_a:.5f} | "
              f"B: train={tl_b:.5f} val={vl_b:.5f} | "
              f"patience={patience_counter}/{MAX_PATIENCE}")

    if patience_counter >= MAX_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# ============================================================
# LOAD BEST WEIGHTS
# ============================================================
autoencoder_a.load_weights('autoencoder_a.weights.h5')
autoencoder_b.load_weights('autoencoder_b.weights.h5')

# ============================================================
# TRAINING CURVES
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history_a_loss, label='Train A')
ax1.plot(history_a_val, label='Val A')
ax1.set_title('Autoencoder A Loss')
ax1.legend()

ax2.plot(history_b_loss, label='Train B')
ax2.plot(history_b_val, label='Val B')
ax2.set_title('Autoencoder B Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=100)
plt.close()

# ============================================================
# QUICK VISUAL CHECK
# ============================================================
sample_a = X_test_a[:5]
recon_a = autoencoder_a.predict(sample_a)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    axes[0, i].imshow(sample_a[i])
    axes[0, i].set_title('Original')
    axes[0, i].axis('off')
    axes[1, i].imshow(np.clip(recon_a[i], 0, 1))
    axes[1, i].set_title('Reconstructed')
    axes[1, i].axis('off')
plt.suptitle('Autoencoder A Reconstruction')
plt.tight_layout()
plt.savefig('reconstruction_check.png', dpi=100)
plt.close()

# ============================================================
# SAVE FULL MODELS FOR NOTEBOOK 3
# ============================================================
encoder.save_weights('encoder.weights.h5')
decoder_a.save_weights('decoder_a.weights.h5')
decoder_b.save_weights('decoder_b.weights.h5')

print("\nTraining complete. Saved weights:")
print(f"  encoder.weights.h5")
print(f"  decoder_a.weights.h5")
print(f"  decoder_b.weights.h5")
print(f"  Best val_loss A: {best_val_a:.5f}")
print(f"  Best val_loss B: {best_val_b:.5f}")