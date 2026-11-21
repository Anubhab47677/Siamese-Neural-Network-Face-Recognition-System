"""
face_recognition_system.py
--------------------------
Real-time face verification and group-photo authentication using the Siamese Neural Network.
Supports single-person verification, crowd/group photo multi-face identification,
and adaptive multi-reference gallery matching.
"""

import os
import cv2
import numpy as np
from utils import MODEL_PATH, DB_PATH, load_embedder, load_config, embed_faces
from face_detection import FaceDetector, draw_boxes
import database as db

# Colors for UI drawing (BGR)
GREEN = (40, 200, 40)
RED = (40, 40, 220)
ORANGE = (30, 140, 255)
CYAN = (220, 180, 40)


class FaceRecognizer:
    def __init__(self, model_path=MODEL_PATH, db_path=DB_PATH, threshold=None,
                 detector=None, embedder=None, k=3, matching_strategy="min"):
        """
        matching_strategy:
          - 'min': (Recommended for group photos & varied expressions) finds closest matching reference pose.
          - 'top_k': takes the mean distance of the k nearest reference embeddings.
          - 'adaptive': 70% best pose + 30% top-k mean for stability.
        """
        self.embedder = embedder or load_embedder(model_path)
        self.detector = detector or FaceDetector(method="auto")
        self.db_path = db_path
        self.k = k
        self.matching_strategy = matching_strategy

        # Load threshold from config or default to 0.6506 (assignment baseline)
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            cfg = load_config()
            self.threshold = float(cfg.get("threshold", 0.6506))

        self.refresh_users()

    def refresh_users(self):
        """Reload registered users and their 128-d embeddings from SQLite."""
        self.users = db.get_all_embeddings(self.db_path)

    def _user_distance(self, emb, gallery):
        """
        Compute distance between probe face embedding and a user's reference gallery.
        Returns (distance, best_sample_idx)
        """
        if len(gallery) == 0:
            return 999.0, -1

        dists = np.linalg.norm(gallery - emb, axis=1)
        best_idx = int(np.argmin(dists))
        sorted_d = np.sort(dists)

        if self.matching_strategy == "min":
            dist = float(sorted_d[0])
        elif self.matching_strategy == "top_k":
            dist = float(sorted_d[:min(self.k, len(sorted_d))].mean())
        elif self.matching_strategy == "adaptive":
            d_min = float(sorted_d[0])
            d_mean = float(sorted_d[:min(self.k, len(sorted_d))].mean())
            dist = 0.70 * d_min + 0.30 * d_mean
        else:
            dist = float(sorted_d[0])

        return dist, best_idx

    def verify_face(self, face):
        """
        Compare one preprocessed face (112, 92, 1) against all registered identities.
        Returns detailed decision dictionary with closest user, distance, and similarity confidence.
        """
        if not self.users:
            return {
                "user_id": None, "name": "Unknown User", "closest_user": "None",
                "distance": 999.0, "recognized": False, "confidence": "None",
                "distances": {}, "best_sample_idx": -1, "sim_score": 0
            }

        # 1. Generate 128-d embedding from shared CNN
        emb = embed_faces(self.embedder, face)[0]

        # 2. Compare against every registered user's gallery
        user_scores = {}
        best_samples = {}
        for uid, u in self.users.items():
            d, b_idx = self._user_distance(emb, u["embeddings"])
            user_scores[uid] = d
            best_samples[uid] = b_idx

        # 3. Find closest enrolled identity
        best_uid = min(user_scores, key=user_scores.get)
        min_d = user_scores[best_uid]
        is_recognized = min_d < self.threshold

        # 4. Confidence scoring
        ratio = min_d / max(self.threshold, 1e-6)
        if is_recognized and ratio < 0.65:
            conf = "High"
        elif is_recognized and ratio < 0.88:
            conf = "Medium"
        elif is_recognized:
            conf = "Moderate"
        else:
            conf = "Low (Impostor)"

        sim_score = max(0, min(100, int((1.0 - (min_d / (self.threshold * 1.4))) * 100)))

        return {
            "user_id": best_uid if is_recognized else None,
            "name": self.users[best_uid]["name"] if is_recognized else "Unknown User",
            "closest_user": self.users[best_uid]["name"],
            "closest_uid": best_uid,
            "distance": min_d,
            "recognized": is_recognized,
            "confidence": conf,
            "sim_score": sim_score,
            "best_sample_idx": best_samples[best_uid],
            "distances": {self.users[k]["name"]: round(v, 4) for k, v in user_scores.items()}
        }

    def recognize_frame(self, frame):
        """
        Execute full detection and identification pipeline on an input frame.
        Gracefully handles: No Face, Single Face, and Dense Multi-Face Group Photos.
        """
        faces = self.detector.get_all_faces(frame)

        if not faces:
            return {
                "status": "NO_FACE",
                "message": "No face detected. Please ensure faces are well-lit and facing the camera.",
                "faces": [],
                "frame": frame,
                "n_faces": 0,
                "n_recognized": 0
            }

        if not self.users:
            return {
                "status": "NO_USERS",
                "message": "No registered users in database. Please enroll reference users first.",
                "faces": [],
                "frame": frame,
                "n_faces": len(faces),
                "n_recognized": 0
            }

        results = []
        for f in faces:
            r = self.verify_face(f["face"])
            r["box"] = f["box"]
            r["face_crop"] = f["face"]
            results.append(r)

        recognized_members = [r["name"] for r in results if r["recognized"]]
        unregistered_count = sum(1 for r in results if not r["recognized"])
        n_faces = len(results)

        if n_faces > 1:
            status = "MULTIPLE_FACES"
            if recognized_members:
                msg = f"👥 Group Photo: {len(recognized_members)} recognized ({', '.join(set(recognized_members))}), {unregistered_count} guest(s)"
            else:
                msg = f"👥 Group Photo: {n_faces} faces detected — all unregistered guests."
        elif results[0]["recognized"]:
            status = "RECOGNIZED"
            msg = f"ACCESS GRANTED: {results[0]['name']} (Distance: {results[0]['distance']:.3f} | {results[0]['confidence']} Confidence)"
        else:
            status = "UNKNOWN"
            msg = f"ACCESS DENIED: Unknown User (Closest: {results[0]['closest_user']} with d={results[0]['distance']:.3f})"

        return {
            "status": status,
            "message": msg,
            "faces": results,
            "frame": frame,
            "n_faces": n_faces,
            "n_recognized": len(recognized_members)
        }

    def annotate(self, result):
        """Annotate bounding boxes with name tags and top banner."""
        fr = result["frame"]
        if not result["faces"]:
            return draw_boxes(fr, [], banner=(result["message"], RED))

        boxes = [r["box"] for r in result["faces"]]
        labels = []
        colors = []

        for r in result["faces"]:
            if r["recognized"]:
                labels.append(f"{r['name']} (d={r['distance']:.2f})")
                colors.append(GREEN)
            else:
                labels.append(f"Unknown (d={r['distance']:.2f})")
                colors.append(RED)

        if result["status"] == "MULTIPLE_FACES":
            banner_col = CYAN if result["n_recognized"] > 0 else ORANGE
        elif result["status"] == "RECOGNIZED":
            banner_col = GREEN
        else:
            banner_col = RED

        return draw_boxes(fr, boxes, labels, colors, banner=(result["message"], banner_col))
