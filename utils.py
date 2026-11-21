"""
utils.py
--------
Dataset loading, preprocessing, pair generation, CNN construction,
Euclidean distance and contrastive loss for the Siamese face-recognition project.
"""
import os, glob, re, json, itertools
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# ------------------------------------------------------------------ paths / constants
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
USERS_DIR  = os.path.join(BASE_DIR, "registered_users")
DB_PATH    = os.path.join(USERS_DIR, "users.db")
MODEL_PATH = os.path.join(MODEL_DIR, "siamese_nn.h5")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

IMG_H, IMG_W = 112, 92
INPUT_SHAPE = (IMG_H, IMG_W, 1)
EMB_DIM = 128
MARGIN = 1.0
IMG_EXT = (".pgm", ".png", ".jpg", ".jpeg", ".bmp")

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


# ================================================================== TASK 1: data
def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def img_to_array(gray, use_clahe=True):
    """uint8 gray/BGR image of any size -> float32 array (112, 92, 1) scaled to [0, 1].
    Steps: grayscale -> resize(112x92) -> CLAHE -> normalise -> add channel axis."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    if use_clahe:
        gray = _clahe.apply(gray)
    return (gray.astype("float32") / 255.0).reshape(IMG_H, IMG_W, 1)


def load_img(path, use_clahe=True):
    """Read one image file from disk and return the preprocessed array (or None)."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if gray is None else img_to_array(gray, use_clahe)


def get_data(dataset_dir, use_clahe=True, verbose=True):
    """Load the whole dataset (one sub-folder per person).
    Returns: X (N,112,92,1) float32, y (N,) int labels, class_names list, paths list."""
    class_dirs = sorted([d for d in os.listdir(dataset_dir)
                         if os.path.isdir(os.path.join(dataset_dir, d))], key=_natural)
    X, y, paths = [], [], []
    for label, cname in enumerate(class_dirs):
        files = sorted(glob.glob(os.path.join(dataset_dir, cname, "*")), key=_natural)
        for f in files:
            if not f.lower().endswith(IMG_EXT):
                continue
            arr = load_img(f, use_clahe)
            if arr is None:
                continue
            X.append(arr); y.append(label); paths.append(f)
    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int32")
    if verbose:
        print(f"Loaded {len(X)} images | {len(class_dirs)} individuals | "
              f"{len(X)//max(len(class_dirs),1)} images per individual | X{X.shape} y{y.shape}")
    return X, y, class_dirs, paths


def split_by_identity(X, y, paths, n_train=28, n_val=6, seed=42):
    """Split by PERSON (not by image) so test people are never seen during training."""
    ids = np.unique(y)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    groups = {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}
    paths = np.asarray(paths)
    out = {}
    for name, g in groups.items():
        m = np.isin(y, g)
        out[name] = (X[m], y[m], paths[m])
    return out, groups


# ================================================================== TASK 2: pairs
def augment_image(img, rng):
    """Random rotation/zoom/shift/flip/brightness/noise on one (112,92,1) image."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.12))
    M[:, 2] += (rng.uniform(-5, 5), rng.uniform(-6, 6))
    out = cv2.warpAffine(img[..., 0], M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.rand() < 0.5:
        out = out[:, ::-1]
    out = out * rng.uniform(0.75, 1.25) + rng.uniform(-0.12, 0.12)
    out = out + rng.normal(0, 0.02, out.shape)
    return np.ascontiguousarray(np.clip(out, 0, 1).astype("float32")[..., None])


def create_pairs(X, y, augment_copies=0, seed=42):
    """Create a balanced, shuffled set of image pairs.
    label 1 = same person (positive), label 0 = different people (negative).
    augment_copies: extra augmented copies of every pair (use only for training data).
    Returns A, B, labels, ids  (ids = identity of each image in the pair, for testing/plots)."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    idx = {c: np.where(y == c)[0] for c in classes}

    pos = [(i, j) for c in classes for i, j in itertools.combinations(idx[c], 2)]   # all same-person pairs
    neg = []
    while len(neg) < len(pos):                                                    # equal number of negatives
        c1, c2 = rng.choice(classes, 2, replace=False)
        neg.append((rng.choice(idx[c1]), rng.choice(idx[c2])))

    pairs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype="float32")
    ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
    A, B = X[ia], X[ib]
    ids = np.stack([y[ia], y[ib]], axis=1)

    if augment_copies > 0:
        As, Bs, Ls, Is = [A], [B], [labels], [ids]
        for _ in range(augment_copies):
            As.append(np.stack([augment_image(a, rng) for a in A]))
            Bs.append(np.stack([augment_image(b, rng) for b in B]))
            Ls.append(labels); Is.append(ids)
        A, B, labels, ids = np.concatenate(As), np.concatenate(Bs), np.concatenate(Ls), np.concatenate(Is)

    perm = rng.permutation(len(labels))                                           # shuffle
    return A[perm], B[perm], labels[perm], ids[perm]


def show_pairs(A, B, labels, n=5, title="", save=None):
    """Plot n positive and n negative pairs."""
    pos = np.where(labels == 1)[0][:n]; neg = np.where(labels == 0)[0][:n]
    fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 6))
    for r, (group, name, col) in enumerate([(pos, "Positive (label=1)", "green"), (neg, "Negative (label=0)", "red")]):
        for c, i in enumerate(group):
            ax[r, c].imshow(np.hstack([A[i, ..., 0], B[i, ..., 0]]), cmap="gray", vmin=0, vmax=1)
            ax[r, c].set_title(name, color=col, fontsize=9); ax[r, c].axis("off")
    fig.suptitle(title); plt.tight_layout()
    if save: plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.show()


# ================================================================== TASK 3/4/5: model pieces
def create_shared_network(arch="lab", input_shape=INPUT_SHAPE):
    """Shared CNN feature extractor -> 128-d embedding.
    arch='lab'      : exactly the architecture in the assignment
    arch='improved' : deeper + BatchNorm + Dropout (optional bonus)"""
    if arch == "lab":
        net = models.Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(EMB_DIM, activation="sigmoid"),
        ], name="shared_cnn")
    else:
        net = models.Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"), layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"), layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"), layers.MaxPooling2D(),
            layers.Flatten(), layers.Dropout(0.3),
            layers.Dense(EMB_DIM, activation="sigmoid"),
        ], name="shared_cnn")
    return net


def euclidean_distance(vects):
    """Euclidean distance between two embedding batches -> shape (batch, 1)."""
    x, y = vects
    sum_sq = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_sq, 1e-7))


def contrastive_loss(y_true, y_pred):
    """L = y * d^2 + (1 - y) * max(margin - d, 0)^2   (y=1 same person, margin = 1)."""
    y_true = tf.cast(tf.reshape(y_true, (-1, 1)), y_pred.dtype)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(MARGIN - y_pred, 0.0))
    return tf.reduce_mean(y_true * square_pred + (1.0 - y_true) * margin_square)


def accuracy(y_true, y_pred):
    """Pair accuracy: distance < 0.5 -> predicted 'same person'."""
    y_true = tf.cast(tf.reshape(y_true, (-1, 1)), y_pred.dtype)
    pred = tf.cast(y_pred < 0.5, y_pred.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, pred), y_pred.dtype))


CUSTOM_OBJECTS = {"euclidean_distance": euclidean_distance,
                  "contrastive_loss": contrastive_loss, "accuracy": accuracy}


def count_trainable(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


# ================================================================== load / embed helpers
def load_siamese_model(path=MODEL_PATH):
    try:
        return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False, safe_mode=False)
    except TypeError:                       # older Keras without safe_mode
        return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)


def get_embedding_model(siamese):
    """The shared CNN (weights shared by both branches) used as feature extractor."""
    return siamese.get_layer("shared_cnn")


def load_embedder(path=MODEL_PATH):
    return get_embedding_model(load_siamese_model(path))


def embed_faces(embedder, faces):
    """faces: (N,112,92,1) float32 -> (N,128) numpy embeddings."""
    faces = np.asarray(faces, dtype="float32")
    if faces.ndim == 3:
        faces = faces[None]
    return embedder(tf.convert_to_tensor(faces), training=False).numpy()


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(**kw):
    os.makedirs(MODEL_DIR, exist_ok=True)
    cfg = load_config(); cfg.update(kw)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ================================================================== evaluation helpers
def pair_distances(siamese, A, B, batch_size=256):
    return siamese.predict([A, B], batch_size=batch_size, verbose=0).ravel()


def find_best_threshold(d, labels, n=500):
    """Sweep thresholds; predict 'same person' when distance < t. Returns best t and curves."""
    ts = np.linspace(d.min(), d.max(), n)
    accs = np.array([np.mean((d < t) == (labels == 1)) for t in ts])
    i = int(np.argmax(accs))
    return float(ts[i]), float(accs[i]), ts, accs
