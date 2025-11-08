# =============================================
# FF++ C23 → Deepfake 7-clases (original + 6 métodos)
# Pipeline completo Colab (caras PNG, EfficientNetB0)
# =============================================

# ---------- 0) Instalar deps ----------
!pip -q install -U "tensorflow==2.18.0" "mediapipe==0.10.14" "opencv-python-headless==4.10.0.84" \
               "numpy==1.26.4" "pandas==2.2.2" "pyarrow==14.0.2" "tqdm" "scikit-learn==1.5.2" "matplotlib==3.9.2"

# ---------- 1) Imports / Paths / Consts ----------
from pathlib import Path
import os, sys, math, random, json, gc, time
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

print("Python:", sys.version)
print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# Cambia esta ruta si tus videos están en otro lugar
ROOT = Path("/content/ffpp_c23")               # Estructura: /content/ffpp_c23/{original, Deepfakes, ...}/*.mp4
OUT  = Path("/content/data_faces_ffpp")        # Aquí se guardarán caras 224x224 PNG
CKPT_DIR = Path("/content/ffpp_checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Clases (orden fijo)
METHODS = [
    "original",
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]
NUM_CLASSES = len(METHODS)

# Imagen / extracción
IMG_SIZE = (224, 224)
MAX_FRAMES_PER_VIDEO = 2          # 8 caras por video (sube/baja según recursos)
SAMPLE_EVERY_SECONDS  = 1.0       # ~1 fps
PNG_COMPRESSION = 3               # 0-9 (baja = menos compresión)
FACE_CONF = 0.5                   # confiabilidad mínima detección
BBOX_SCALE = 1.2                  # expandir caja (1.0=igual)

# Data
BATCH = 64
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- 2) Utilidades ----------
def list_videos(folder: Path):
    return sorted(folder.glob("*.mp4"))

def split_list(lst, tr=0.7, va=0.15):
    n = len(lst); a = int(n*tr); b = int(n*(tr+va))
    return lst[:a], lst[a:b], lst[b:]

def clamp(v, a, b):
    return max(a, min(b, v))

def _count_images(p: Path, ext=(".png",".jpg",".jpeg")):
    return sum(1 for _ in p.rglob("*") if _.suffix.lower() in ext)

# ---------- 3) Detección de rostro (MediaPipe) + crop ----------
def _build_mp_detector():
    # model_selection=1 para distancias largas; 0 para cortas. Con C23 vídeos variados, 1 va bien.
    return mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=FACE_CONF)

def crop_face_mediapipe(img_bgr, detector, out_size=224, enlarge=BBOX_SCALE):
    """ Devuelve PNG-ready BGR crop (H,W,3) o None si no hay cara """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = detector.process(img_rgb)
    if not res.detections:
        return None

    # tomar la detección con mayor score
    det = max(res.detections, key=lambda d: d.score[0] if d.score else 0.0)
    if not det.score or det.score[0] < FACE_CONF: 
        return None
    bb = det.location_data.relative_bounding_box
    cx, cy = bb.xmin + bb.width/2, bb.ymin + bb.height/2
    bw, bh = bb.width, bb.height
    # cuadrado ampliado
    s = max(bw, bh) * enlarge
    x1 = int((cx - s/2) * w); y1 = int((cy - s/2) * h)
    x2 = int((cx + s/2) * w); y2 = int((cy + s/2) * h)
    x1, y1 = clamp(x1, 0, w-1), clamp(y1, 0, h-1)
    x2, y2 = clamp(x2, 0, w),   clamp(y2, 0, h)
    if x2<=x1 or y2<=y1: 
        return None
    face = img_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return face

def process_one_video(vp: Path, out_dir: Path, max_frames=MAX_FRAMES_PER_VIDEO):
    """ Extrae ~1 fps, recorta cara con MediaPipe, guarda PNG. """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved, failed = 0, 0
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return (str(vp), 0, "no_open")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps * SAMPLE_EVERY_SECONDS)), 1)

    with _build_mp_detector() as fd:
        i, fidx = 0, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0 and saved < max_frames:
                face = crop_face_mediapipe(fr, fd, out_size=IMG_SIZE[0], enlarge=BBOX_SCALE)
                if face is None:
                    failed += 1
                else:
                    outp = out_dir / f"{vp.stem}_f{fidx:06d}.png"
                    cv2.imwrite(str(outp), face, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
                    saved += 1
                fidx += 1
            i += 1
    cap.release()
    status = "ok" if saved>0 else ("no_faces" if failed>0 else "empty")
    return (str(vp), saved, status)

# ---------- 4) Construir splits por MÉTODO (70/15/15) ----------
videos_by_method = {m: list_videos(ROOT/m) for m in METHODS}
for m in METHODS:
    random.shuffle(videos_by_method[m])

SPLITS_METHODS = {
    "train": {m: split_list(videos_by_method[m])[0] for m in METHODS},
    "val":   {m: split_list(videos_by_method[m])[1] for m in METHODS},
    "test":  {m: split_list(videos_by_method[m])[2] for m in METHODS},
}
print({sp: {m: len(SPLITS_METHODS[sp][m]) for m in METHODS} for sp in ["train","val","test"]})

# ---------- 5) Extracción de caras en paralelo por videos ----------
from concurrent.futures import ProcessPoolExecutor, as_completed

def save_frames_parallel(videos, out_dir: Path, max_frames_per_video=MAX_FRAMES_PER_VIDEO, max_workers=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process_one_video, vp, out_dir, max_frames_per_video) for vp in videos]
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"Procesando → {out_dir.name}", leave=False):
            results.append(f.result())
    ok = sum(1 for _, n, s in results if s == "ok" and n > 0)
    skipped = [r for r in results if r[2] != "ok"]
    if skipped:
        print(f"Saltados/errores: {len(skipped)} (p.ej., {skipped[0]})")
    return results

for split in ["train","val","test"]:
    total = 0
    for m in METHODS:
        vids = SPLITS_METHODS[split][m]
        dest = OUT / split / m
        print(f"[{split}/{m}] videos={len(vids)} → {dest}")
        _ = save_frames_parallel(vids, dest, max_frames_per_video=MAX_FRAMES_PER_VIDEO, max_workers=4)
        total += len(vids)
    print(f"Total {split}: {total} videos")

print("Conteo imágenes por split/clase:")
for sp in ["train","val","test"]:
    counts = {m: _count_images(OUT/sp/m) for m in METHODS}
    print(sp, counts)

# ---------- 6) tf.data (imagen→tensor) ----------
def build_ds(split, batch=BATCH, cache=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        OUT / split,
        labels="inferred",
        label_mode="int",
        class_names=METHODS,      # orden fijo
        image_size=IMG_SIZE,
        batch_size=batch,
        shuffle=True,
        seed=SEED,
        interpolation="bilinear"
    )
    if cache:
        ds = ds.cache()
    return ds.prefetch(AUTOTUNE)

ds_train = build_ds("train", cache=False)   # no cache en train
ds_val   = build_ds("val",   cache=True)    # cache val
ds_test  = build_ds("test",  cache=True)

# ---------- 7) Cálculo de class_weight (7 clases) ----------
cls_counts = {i: _count_images(OUT/"train"/METHODS[i]) for i in range(NUM_CLASSES)}
tot = sum(cls_counts.values())
class_weight = {i: (tot / (NUM_CLASSES * max(1, cls_counts[i]))) for i in range(NUM_CLASSES)}
print("class_weight:", class_weight)

# ---------- 8) Capa de augment de recompresión JPEG ----------
class RandomJPEG(layers.Layer):
    def __init__(self, qmin=60, qmax=95, **kwargs):
        super().__init__(**kwargs); self.qmin, self.qmax = qmin, qmax
    def call(self, x, training=None):
        if training is False: return x
        q = tf.random.uniform([], self.qmin, self.qmax, dtype=tf.int32)
        x8 = tf.cast(tf.clip_by_value(x*255.0, 0.0, 255.0), tf.uint8)   # x ∈ [0,1]
        bytes_ = tf.io.encode_jpeg(x8, quality=q, chroma_downsampling=True)
        x2 = tf.io.decode_jpeg(bytes_, channels=3)
        return tf.cast(x2, tf.float32) / 255.0

# ---------- 9) Modelo EfficientNetB0 (warmup + fine-tune) ----------
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

def build_model_efficientnetb0():
    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Rescaling(1./255)(inp)             # a [0,1]
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = RandomJPEG(60,95)(x)                      # sigue en [0,1]
    x = layers.Lambda(lambda t: t*255.0)(x)       # volver a 0..255 para preprocess oficial
    x = layers.Lambda(eff_preprocess)(x)

    base = EfficientNetB0(include_top=False, weights="imagenet")
    base.trainable = False                        # warmup congelado
    y = base(x, training=False)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dropout(0.30)(y)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="out")(y)
    model = keras.Model(inp, out)
    return model, base

model, base = build_model_efficientnetb0()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc"),
             keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")]
)
model.summary()

# ---------- 10) Callbacks ----------
ckpt_every_epoch = str(CKPT_DIR / "weights_epoch_{epoch:03d}.weights.h5")
ckpt_best        = str(CKPT_DIR / "best_by_valACC.keras")
csv_log          = str(CKPT_DIR / "training_log.csv")

cbs = [
    keras.callbacks.ModelCheckpoint(
        filepath=ckpt_every_epoch, save_weights_only=True, save_freq="epoch", verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=ckpt_best, save_weights_only=False, monitor="val_acc", mode="max",
        save_best_only=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=4, restore_best_weights=True, verbose=1),
    keras.callbacks.CSVLogger(csv_log, append=True),
]

# ---------- 11) Entrenamiento: fase 1 (warmup, base congelada) ----------
WARMUP_EPOCHS = 5
history1 = model.fit(
    ds_train, validation_data=ds_val,
    epochs=WARMUP_EPOCHS,
    callbacks=cbs,
    class_weight=class_weight,
    verbose=1
)

# ---------- 12) Fine-tune: descongela parte final ----------
# Descongelar últimas ~50% capas (ajústalo si tienes más GPU)
n_layers = len(base.layers)
for li, lyr in enumerate(base.layers):
    lyr.trainable = (li >= n_layers//2)

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # LR bajo para fine-tune
    loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc"),
             keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")]
)

FINE_EPOCHS = 10
history2 = model.fit(
    ds_train, validation_data=ds_val,
    epochs=FINE_EPOCHS,
    callbacks=cbs,
    class_weight=class_weight,
    verbose=1
)

# ---------- 13) Cargar mejor modelo por val_acc ----------
best_model = keras.models.load_model(ckpt_best)
print("Cargado mejor modelo:", ckpt_best)

# ---------- 14) Evaluación en test ----------
# Recolectar probabilidades y etiquetas
y_true, y_pred, y_prob = [], [], []
for bx, by in ds_test:
    p = best_model.predict(bx, verbose=0)
    y_prob.append(p)
    y_true.append(by.numpy())
    y_pred.append(np.argmax(p, axis=1))

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
y_prob = np.concatenate(y_prob)

# Reporte por clase
report = classification_report(y_true, y_pred, target_names=METHODS, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(str(CKPT_DIR / "classification_report_test.csv"))
print(df_report)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
fig = plt.figure(figsize=(7,6))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix (test)")
plt.colorbar()
ticks = np.arange(NUM_CLASSES)
plt.xticks(ticks, METHODS, rotation=45, ha='right')
plt.yticks(ticks, METHODS)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
plt.savefig(str(CKPT_DIR / "confusion_matrix_test.png"), dpi=150)
plt.show()

# AUC one-vs-rest (macro)
try:
    # binariza y calcula ROC-AUC macro/weighted (puede fallar si una clase no aparece)
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auc_macro = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    auc_weighted = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
    print(f"ROC-AUC macro (ovr): {auc_macro:.4f} | weighted: {auc_weighted:.4f}")
except Exception as e:
    print("AUC OVR no calculable:", e)

# Guardar el mejor modelo final
best_path = str(CKPT_DIR / "best_final.keras")
best_model.save(best_path)
print("Modelo guardado en:", best_path)

# ---------- 15) Pequeño resumen ----------
print("\nResumen:")
print(" - Caras en:", OUT)
print(" - Checkpoints en:", CKPT_DIR)
print(" - Reporte CSV:", str(CKPT_DIR / "classification_report_test.csv"))
print(" - Confusion matrix PNG:", str(CKPT_DIR / "confusion_matrix_test.png"))
