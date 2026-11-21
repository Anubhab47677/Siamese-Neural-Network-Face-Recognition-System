"""
onboarding.py
-------------
User enrolment: capture reference faces, crop/preprocess them, create 128-d embeddings
with the trained Siamese network's shared CNN and store the user.
Includes Automatic Consistency Verification to prevent identity contamination.
"""

import os
import time
import cv2
import numpy as np
from utils import DB_PATH, MODEL_PATH, load_embedder, embed_faces
from face_detection import FaceDetector, capture_frames_colab, frames_from_upload, draw_boxes
import database as db


def register_user(user_id, name, frames, embedder=None, detector=None, db_path=DB_PATH,
                  min_faces=3, overwrite=False, save_images=True, verbose=True):
    """
    Register a user from a list of BGR frames.
    - Frames with no face or multiple faces are skipped to prevent enrolling the wrong person.
    - Verifies internal gallery consistency so photos of different people cannot be mixed into one identity.
    """
    user_id, name = str(user_id).strip(), str(name).strip()
    if not user_id or not name:
        return {"success": False, "message": "User ID and name are required."}
    if db.user_exists(user_id, db_path) and not overwrite:
        return {"success": False, "message": f"User ID '{user_id}' already exists. Use another ID or enable overwrite."}

    embedder = embedder or load_embedder(MODEL_PATH)
    detector = detector or FaceDetector()

    faces, crops, no_face, multi = [], [], 0, 0
    for fr in frames:
        boxes = detector.detect(fr)
        if len(boxes) == 0:
            no_face += 1
            continue
        if len(boxes) > 1:
            multi += 1
            continue
        f = detector.crop(fr, boxes[0])
        faces.append(f)
        crops.append((f[..., 0] * 255).astype("uint8"))

    if len(faces) < min_faces:
        msg = (f"Registration failed: only {len(faces)} usable face(s) found "
               f"({no_face} frames without a face, {multi} with multiple faces); need at least {min_faces}.")
        if verbose:
            print(msg)
        return {"success": False, "message": msg, "usable": len(faces), "no_face": no_face, "multi_face": multi}

    emb = embed_faces(embedder, np.stack(faces))  # (n, 128)

    # -------------------------------------------------------------------------
    # Gallery Consistency Check: prevent mixing different individuals into one ID
    # -------------------------------------------------------------------------
    if len(faces) >= 2:
        n = len(faces)
        # Find medoid (the most representative face among uploaded samples)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(emb[i] - emb[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        medoid_idx = int(np.argmin(dist_matrix.sum(axis=1)))
        dists_to_medoid = dist_matrix[medoid_idx]

        # Samples closer than 0.65 match the same identity
        consistent_mask = dists_to_medoid < 0.65
        n_consistent = int(np.sum(consistent_mask))

        if n_consistent < len(faces):
            outlier_count = len(faces) - n_consistent
            if n_consistent >= min_faces:
                # Filter out the inconsistent faces (keep only the consistent person)
                faces = [f for i, f in enumerate(faces) if consistent_mask[i]]
                crops = [c for i, c in enumerate(crops) if consistent_mask[i]]
                emb = emb[consistent_mask]
                if verbose:
                    print(f"[Onboarding] Note: Removed {outlier_count} inconsistent photo(s) that belonged to a different person.")
            else:
                msg = (f"Registration rejected: Uploaded photos contain faces of different people "
                       f"(internal distance {dist_matrix.max():.2f} > 0.65). "
                       f"Please ensure all photos uploaded belong to '{name}'.")
                if verbose:
                    print(msg)
                return {"success": False, "message": msg, "usable": 0, "no_face": no_face, "multi_face": multi}

    db.add_user(user_id, name, emb, n_images=len(faces), db_path=db_path, overwrite=overwrite)

    if save_images:
        folder = os.path.join(os.path.dirname(db_path), user_id)
        os.makedirs(folder, exist_ok=True)
        # Clean existing crops if overwriting
        for old_file in os.listdir(folder):
            if old_file.endswith(".png"):
                try:
                    os.remove(os.path.join(folder, old_file))
                except Exception:
                    pass
        for k, c in enumerate(crops):
            cv2.imwrite(os.path.join(folder, f"face_{k+1:02d}.png"), c)

    msg = f"✅ User '{name}' (ID: {user_id}) registered successfully with {len(faces)} verified face samples."
    if verbose:
        print(msg)
    return {
        "success": True,
        "message": msg,
        "usable": len(faces),
        "no_face": no_face,
        "multi_face": multi,
        "embeddings": emb,
        "crops": crops
    }


def enroll_via_colab_webcam(user_id, name, embedder=None, detector=None, n=10, interval_ms=700, **kw):
    frames = capture_frames_colab(n=n, interval_ms=interval_ms, label=f"📸 Start capturing {n} photos for {name}")
    return register_user(user_id, name, frames, embedder, detector, **kw)


def enroll_via_uploads(user_id, name, embedder=None, detector=None, **kw):
    return register_user(user_id, name, frames_from_upload(), embedder, detector, **kw)


def enroll_via_local_webcam(user_id, name, embedder=None, detector=None, n=10, camera_index=0, **kw):
    detector = detector or FaceDetector()
    cap = cv2.VideoCapture(camera_index)
    frames, last = [], 0
    while cap.isOpened() and len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        boxes = detector.detect(fr)
        cv2.imshow("Enrolment (q = quit)", draw_boxes(fr, boxes[:1], banner=(f"Captured {len(frames)}/{n}", (0, 255, 0))))
        if len(boxes) == 1 and time.time() - last > 0.7:
            frames.append(fr.copy())
            last = time.time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return register_user(user_id, name, frames, embedder, detector, **kw)
