# =============================================================================
# Drowsiness Detection System (DDD) - CNN Model with Data Augmentation
# Improved version : EfficientNetB3, 224×224, CBAM, Mixup, two-phase training
# =============================================================================

# ── 1. Imports ────────────────────────────────────────────────────────────────

import os
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     Input, Conv2D, Multiply, Add, Reshape,
                                     Permute, Activation, Lambda)
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight
import gc


# ── 2. Configuration ──────────────────────────────────────────────────────────

TRAIN_PATH       = r'C:\Users\HP\Downloads\train'
AUGMENTED_PATH   = r'C:\Users\HP\Downloads\train_augmented'
CLASS_NAMES      = ['Closed', 'Open', 'no_yawn', 'yawn']
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 8           # reduced to avoid OOM on CPU
EPOCHS_PHASE1    = 15          # frozen backbone
EPOCHS_PHASE2    = 20          # fine-tuning top-50 layers
RANDOM_SEED      = 42
AUGMENT_FACTOR   = 2
OUTPUT_DIR       = r'C:\Users\HP\Downloads\train\output_plots'
FINE_TUNE_LAYERS = 50          # number of top layers to unfreeze in Phase 2
MIXUP_ALPHA      = 0.2         # mixup blending strength

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# =============================================================================
# PARTIE A : Affichage du nombre d'images AVANT augmentation
# =============================================================================

def count_images_per_class(base_path: str, class_names: list) -> dict:
    counts = {}
    for cls in class_names:
        folder = os.path.join(base_path, cls)
        if os.path.exists(folder):
            counts[cls] = len([f for f in os.listdir(folder)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        else:
            counts[cls] = 0
    return counts


def plot_counts(counts: dict, title: str, filename: str):
    names  = list(counts.keys())
    values = list(counts.values())
    total  = sum(values)
    colors = ['steelblue', 'orange', 'green', 'red']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=colors)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', fontweight='bold')
    plt.title(f"{title}  (Total = {total})")
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'images")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=100)
    print(f"  [OK] Graphique sauvegarde : {filename}")
    plt.close()


print("=" * 60)
print("  NOMBRE D'IMAGES AVANT AUGMENTATION")
print("=" * 60)

original_counts = count_images_per_class(TRAIN_PATH, CLASS_NAMES)
for cls, cnt in original_counts.items():
    print(f"  {cls:10s} : {cnt} images")
print(f"  {'TOTAL':10s} : {sum(original_counts.values())} images\n")

plot_counts(original_counts, "Distribution AVANT augmentation", '01_distribution_avant.png')


# =============================================================================
# PARTIE B : Augmentation des donnees et sauvegarde dans un dossier
# =============================================================================

def create_augmented_dataset(src_path: str, dst_path: str, class_names: list,
                             img_size: tuple, target_count: int):
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    for cls in class_names:
        src_folder = os.path.join(src_path, cls)
        dst_folder = os.path.join(dst_path, cls)
        os.makedirs(dst_folder, exist_ok=True)

        files = [f for f in os.listdir(src_folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        n_orig = len(files)

        augment_factor = max(0, int(np.ceil((target_count - n_orig) / max(n_orig, 1))))
        print(f"  Traitement '{cls}' : {n_orig} images -> facteur x{augment_factor} -> ", end="", flush=True)

        augmented_count = 0

        for fname in files:
            img_path = os.path.join(src_folder, fname)
            img = load_img(img_path, target_size=img_size, color_mode='rgb')
            img_array = img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)

            base_name = os.path.splitext(fname)[0]
            orig_save_path = os.path.join(dst_folder, f"{base_name}_orig.jpg")
            cv2.imwrite(orig_save_path, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            augmented_count += 1

            prefix = f"{base_name}_aug"
            i = 0
            for batch in datagen.flow(img_array, batch_size=1):
                aug_img = batch[0].astype(np.uint8)
                aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(dst_folder, f"{prefix}_{i}.jpg")
                cv2.imwrite(save_path, aug_bgr)
                augmented_count += 1
                i += 1
                if i >= augment_factor:
                    break

        print(f"{augmented_count} images (cible : {target_count})")

    print("  Sauvegarde terminee !")


print("=" * 60)
print("  CREATION DU DATASET AUGMENTE")
print("=" * 60)

if os.path.exists(AUGMENTED_PATH):
    print(f"  Le dossier '{AUGMENTED_PATH}' existe deja - augmentation ignoree.")
    print("  Supprimez-le si vous voulez relancer l'augmentation.\n")
else:
    max_count = max(original_counts.values())
    target_per_class = max_count * (AUGMENT_FACTOR + 1)
    print(f"  Dossier de destination : {AUGMENTED_PATH}")
    print(f"  Cible par classe       : {target_per_class} images (equilibrage)\n")
    create_augmented_dataset(TRAIN_PATH, AUGMENTED_PATH, CLASS_NAMES,
                             IMG_SIZE, target_per_class)
    print()


# =============================================================================
# PARTIE C : Affichage du nombre d'images APRES augmentation
# =============================================================================

print("=" * 60)
print("  NOMBRE D'IMAGES APRES AUGMENTATION")
print("=" * 60)

augmented_counts = count_images_per_class(AUGMENTED_PATH, CLASS_NAMES)
for cls, cnt in augmented_counts.items():
    orig = original_counts[cls]
    print(f"  {cls:10s} : {cnt} images  (etait {orig}, +{cnt - orig} ajoutees)")
print(f"  {'TOTAL':10s} : {sum(augmented_counts.values())} images\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['steelblue', 'orange', 'green', 'red']

for ax, counts, title in zip(axes,
                              [original_counts, augmented_counts],
                              ["AVANT augmentation", "APRES augmentation"]):
    names  = list(counts.keys())
    values = list(counts.values())
    bars = ax.bar(names, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title(f"{title}  (Total = {sum(values)})")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'images")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_comparaison_avant_apres.png'), dpi=100)
print("  -> Graphique sauvegarde : 02_comparaison_avant_apres.png")
plt.close()


# =============================================================================
# PARTIE D : Chargement du dataset augmente
# =============================================================================

def load_dataset(base_path: str, class_names: list, img_size: tuple):
    images, labels = [], []
    for idx, label in enumerate(class_names):
        folder_path = os.path.join(base_path, label)
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(idx)
    return np.array(images), np.array(labels)


print("Chargement du dataset augmente ...")
images, labels = load_dataset(AUGMENTED_PATH, CLASS_NAMES, IMG_SIZE)
print(f"Images shape : {images.shape}")
print(f"Labels shape : {labels.shape}")


# =============================================================================
# PARTIE E : Visualisation d'exemples augmentes
# =============================================================================

def show_augmented_examples(base_path: str, class_names: list, n_per_class: int = 4):
    fig, axes = plt.subplots(len(class_names), n_per_class, figsize=(14, 12))
    for row, cls in enumerate(class_names):
        folder = os.path.join(base_path, cls)
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        samples = random.sample(files, min(n_per_class, len(files)))
        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(samples):
                img = cv2.imread(os.path.join(folder, samples[col]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            ax.set_title(cls if col == 0 else "", fontsize=10)
            ax.axis('off')
    plt.suptitle("Exemples d'images apres augmentation", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_exemples_augmentes.png'), dpi=100)
    print("  -> Graphique sauvegarde : 03_exemples_augmentes.png")
    plt.close()


show_augmented_examples(AUGMENTED_PATH, CLASS_NAMES)


# =============================================================================
# PARTIE F : Preparation des donnees (split + encodage)
# =============================================================================

x_train, x_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.4, shuffle=True, random_state=RANDOM_SEED
)
del images, labels; gc.collect()

x_test, x_val, y_test, y_val = train_test_split(
    x_temp, y_temp, test_size=0.5, shuffle=True, random_state=RANDOM_SEED
)
del x_temp, y_temp; gc.collect()

num_classes = len(CLASS_NAMES)
y_train = to_categorical(y_train, num_classes=num_classes)
y_val   = to_categorical(y_val,   num_classes=num_classes)
y_test  = to_categorical(y_test,  num_classes=num_classes)

print(f"Train      : {x_train.shape}, {y_train.shape}")
print(f"Validation : {x_val.shape},   {y_val.shape}")
print(f"Test       : {x_test.shape},   {y_test.shape}")

y_train_labels = np.argmax(y_train, axis=1)
class_weights_arr = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train_labels)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}
print(f"  Poids de classe : {class_weight_dict}")


# =============================================================================
# PARTIE G : Mixup data augmentation (on-the-fly)
# =============================================================================

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def batch_generator(x, y, batch_size, shuffle=True, use_mixup=True):
    n = len(x)
    while True:
        idx = np.arange(n)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            bx = preprocess_input(x[batch_idx].astype(np.float32))
            by = y[batch_idx]
            if use_mixup and shuffle:
                bx, by = mixup_data(bx, by, alpha=MIXUP_ALPHA)
            yield bx, by

train_gen = batch_generator(x_train, y_train, BATCH_SIZE, shuffle=True, use_mixup=True)
val_gen   = batch_generator(x_val,   y_val,   BATCH_SIZE, shuffle=False, use_mixup=False)

steps_train = len(x_train) // BATCH_SIZE
steps_val   = len(x_val)   // BATCH_SIZE


# =============================================================================
# PARTIE H : CBAM Attention Module
# =============================================================================

def cbam_block(feature_map, reduction_ratio=16):
    """Convolutional Block Attention Module (CBAM)."""
    channel = feature_map.shape[-1]

    # --- Channel Attention ---
    # Shared MLP on avg-pool and max-pool
    avg_pool = tf.reduce_mean(feature_map, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(feature_map, axis=[1, 2], keepdims=True)

    shared_dense_1 = Dense(channel // reduction_ratio, activation='relu')
    shared_dense_2 = Dense(channel, activation='sigmoid')

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))
    channel_att = Add()([avg_out, max_out])
    channel_att = Multiply()([feature_map, channel_att])

    # --- Spatial Attention ---
    avg_spatial = tf.reduce_mean(channel_att, axis=-1, keepdims=True)
    max_spatial = tf.reduce_max(channel_att, axis=-1, keepdims=True)
    concat = tf.concat([avg_spatial, max_spatial], axis=-1)
    spatial_att = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    refined = Multiply()([channel_att, spatial_att])

    return refined


# =============================================================================
# PARTIE I : Architecture du modele CNN (EfficientNetB3 + CBAM)
# =============================================================================

def build_model(input_shape: tuple, num_classes: int):
    inputs = Input(shape=input_shape)

    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False

    features = base_model.output                                  # (H, W, C)

    # CBAM attention on extracted features
    att_features = cbam_block(features, reduction_ratio=16)

    x = GlobalAveragePooling2D()(att_features)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, base_model


input_shape = x_train.shape[1:]
model, base_model = build_model(input_shape, num_classes)
model.summary()


# =============================================================================
# PARTIE J : Phase 1 - Entrainement (backbone gele)
# =============================================================================

print("\n" + "=" * 60)
print("  PHASE 1 : Transfer Learning (backbone gele)")
print("=" * 60)

callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

history_phase1 = model.fit(
    train_gen, steps_per_epoch=steps_train,
    epochs=EPOCHS_PHASE1,
    validation_data=val_gen, validation_steps=steps_val,
    callbacks=callbacks_phase1,
    class_weight=class_weight_dict
)


# =============================================================================
# PARTIE K : Phase 2 - Fine-tuning (degele top-50 couches)
# =============================================================================

print("\n" + "=" * 60)
print(f"  PHASE 2 : Fine-tuning (top {FINE_TUNE_LAYERS} couches degelées)")
print("=" * 60)

# Unfreeze top N layers
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

# Re-compile with lower LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Couches entrainables : {sum(1 for l in model.layers if l.trainable)} / {len(model.layers)}")

callbacks_phase2 = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]

history_phase2 = model.fit(
    train_gen, steps_per_epoch=steps_train,
    epochs=EPOCHS_PHASE2,
    validation_data=val_gen, validation_steps=steps_val,
    callbacks=callbacks_phase2,
    class_weight=class_weight_dict
)


# =============================================================================
# PARTIE L : Evaluation sur le jeu de test
# =============================================================================

test_loss, test_accuracy = model.evaluate(
    batch_generator(x_test, y_test, BATCH_SIZE, shuffle=False, use_mixup=False),
    steps=len(x_test) // BATCH_SIZE + 1
)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# =============================================================================
# PARTIE M : Courbes d'apprentissage (combinees Phase 1 + Phase 2)
# =============================================================================

def merge_histories(h1, h2):
    merged = {}
    for k in h1.history:
        merged[k] = h1.history[k] + h2.history.get(k, [])
    return merged


def plot_training_history(h1, h2, output_dir: str):
    merged = merge_histories(h1, h2)
    phase1_len = len(h1.history['accuracy'])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(merged['accuracy'],     label='Train')
    plt.plot(merged['val_accuracy'], label='Validation')
    plt.axvline(x=phase1_len - 1, color='gray', linestyle='--', label='Fine-tune start')
    plt.title('Accuracy')
    plt.xlabel('Epoque')
    plt.ylabel('Precision')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(merged['loss'],     label='Train')
    plt.plot(merged['val_loss'], label='Validation')
    plt.axvline(x=phase1_len - 1, color='gray', linestyle='--', label='Fine-tune start')
    plt.title('Loss')
    plt.xlabel('Epoque')
    plt.ylabel('Perte')
    plt.legend(loc='upper left')

    plt.tight_layout()
    path = os.path.join(output_dir, '04_courbes_apprentissage.png')
    plt.savefig(path, dpi=100)
    print(f"  -> Graphique sauvegarde : 04_courbes_apprentissage.png")
    plt.close()


plot_training_history(history_phase1, history_phase2, OUTPUT_DIR)


# =============================================================================
# PARTIE N : Matrice de confusion + rapport de classification
# =============================================================================

def plot_confusion_matrix(model, x_test, y_test, class_names, output_dir: str):
    y_true = np.argmax(y_test, axis=1)
    y_pred_list = []
    for i in range(0, len(x_test), BATCH_SIZE):
        batch = preprocess_input(x_test[i:i + BATCH_SIZE].astype(np.float32))
        y_pred_list.append(model.predict(batch, verbose=0))
    y_pred_probs = np.concatenate(y_pred_list, axis=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title('Matrice de Confusion - DDD Detection', fontsize=14)
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Realite', fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, '05_matrice_confusion.png')
    plt.savefig(path, dpi=100)
    print(f"  -> Graphique sauvegarde : 05_matrice_confusion.png")
    plt.close()

    print("\nRapport de classification :")
    print(classification_report(y_true, y_pred, target_names=class_names))


plot_confusion_matrix(model, x_test, y_test, CLASS_NAMES, OUTPUT_DIR)


# =============================================================================
# PARTIE O : Exemples de predictions
# =============================================================================

def show_random_predictions(model, x_data, y_data, class_names, output_dir: str, n=6):
    indices       = random.sample(range(len(x_data)), n)
    samples       = x_data[indices]
    actual_labels = np.argmax(y_data[indices], axis=1)
    preds         = model.predict(preprocess_input(samples.astype(np.float32)))
    predicted_labels = np.argmax(preds, axis=1)

    fig, axes = plt.subplots(2, n // 2, figsize=(12, 7))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(samples[i].astype(np.uint8))
        color = 'green' if actual_labels[i] == predicted_labels[i] else 'red'
        ax.set_title(
            f"Predit : {class_names[predicted_labels[i]]}\n"
            f"Reel   : {class_names[actual_labels[i]]}",
            color=color
        )
        ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, '06_predictions_exemples.png')
    plt.savefig(path, dpi=100)
    print(f"  -> Graphique sauvegarde : 06_predictions_exemples.png")
    plt.close()


show_random_predictions(model, x_test, y_test, CLASS_NAMES, OUTPUT_DIR)


# =============================================================================
# PARTIE P : Sauvegarde du modele
# =============================================================================

MODEL_PATH = os.path.join(OUTPUT_DIR, 'DDD_model_augmented.weights.h5')
model.save_weights(MODEL_PATH)
print(f"\nPoids sauvegardes -> {MODEL_PATH}")

import json
CONFIG_PATH = os.path.join(OUTPUT_DIR, 'DDD_model_augmented_config.json')
with open(CONFIG_PATH, 'w') as f:
    json.dump(json.loads(model.to_json()), f, indent=2)
print(f"Architecture sauvegardee -> {CONFIG_PATH}")


# =============================================================================
# PARTIE Q : Resume comparatif final
# =============================================================================

print("\n" + "=" * 60)
print("  RESUME FINAL")
print("=" * 60)
print(f"  Backbone              : EfficientNetB3")
print(f"  Resolution            : {IMG_SIZE[0]}x{IMG_SIZE[1]}")
print(f"  CBAM Attention        : Oui")
print(f"  Mixup (alpha)         : {MIXUP_ALPHA}")
print(f"  Fine-tune layers      : top {FINE_TUNE_LAYERS}")
print(f"  Images originales     : {sum(original_counts.values())}")
print(f"  Images apres augment. : {sum(augmented_counts.values())}")
print(f"  Ratio d'augmentation  : x{AUGMENT_FACTOR + 1}")
print(f"  Phase 1 epochs        : {EPOCHS_PHASE1}")
print(f"  Phase 2 epochs        : {EPOCHS_PHASE2}")
print(f"  Test Accuracy         : {test_accuracy:.4f}")
print(f"  Test Loss             : {test_loss:.4f}")
print(f"  Modele sauvegarde     : {MODEL_PATH}")
print(f"  Graphiques            : {OUTPUT_DIR}/")
print("=" * 60)