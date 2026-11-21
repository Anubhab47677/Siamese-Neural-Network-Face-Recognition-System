"""
siamese_nn.py
-------------
Construction, compilation, training and saving of the Siamese Neural Network.
Run directly:  python siamese_nn.py <path-to-att_faces>
"""
import os, sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from utils import *


def build_siamese(arch="lab"):
    """Two inputs -> SAME shared CNN (shared weights) -> Euclidean distance (Lambda layer)."""
    base = create_shared_network(arch)
    in_a = layers.Input(shape=INPUT_SHAPE, name="input_image_1")
    in_b = layers.Input(shape=INPUT_SHAPE, name="input_image_2")
    feat_a, feat_b = base(in_a), base(in_b)
    dist = layers.Lambda(euclidean_distance, output_shape=(1,), name="euclidean_distance")([feat_a, feat_b])
    return models.Model([in_a, in_b], dist, name="siamese_network"), base


def compile_siamese(model, lr=1e-3):
    model.compile(loss=contrastive_loss, optimizer=keras.optimizers.Adam(lr), metrics=[accuracy])
    return model


def train_siamese(model, train, val, epochs=10, batch_size=128, patience=None, verbose=1):
    """train/val = (A, B, labels). patience -> enables EarlyStopping + LR reduction."""
    cbs = []
    if patience:
        cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 3),
                                                 min_lr=1e-5, verbose=1)]
    h = model.fit([train[0], train[1]], train[2],
                  validation_data=([val[0], val[1]], val[2]),
                  batch_size=batch_size, epochs=epochs, callbacks=cbs, verbose=verbose)
    return h.history


def save_siamese(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    return path


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "dataset", "att_faces")
    X, y, names, paths = get_data(data_dir)
    sp, _ = split_by_identity(X, y, paths)
    tr = create_pairs(sp["train"][0], sp["train"][1], augment_copies=2, seed=1)
    va = create_pairs(sp["val"][0], sp["val"][1], seed=2)
    model, _ = build_siamese()
    compile_siamese(model)
    train_siamese(model, tr[:3], va[:3], epochs=10)
    train_siamese(model, tr[:3], va[:3], epochs=40, patience=8)
    print("Saved:", save_siamese(model))
