"""
face_detection.py
-----------------
Face detection, cropping and drawing utilities with Dual-Engine Architecture:
1. YuNet Deep Learning Detector (Fast, highly accurate, detects small/angled faces in group photos)
2. Viola-Jones Haar Cascade Classifier (Classical baseline from Lab Assignment 06)
Includes landmark geometry verification to reject hands, clothing, and false detections.
"""

import os
import re
import json
import base64
import cv2
import numpy as np
from utils import IMG_H, IMG_W, img_to_array

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_PROFILE_PATH = cv2.data.haarcascades + "haarcascade_profileface.xml"
YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet.onnx")

CROP_SCALE = 1.6  # Context padding around face (hair/forehead/chin); matches AT&T framing


def _strip_emojis(text):
    """Strip non-ASCII characters and emojis to avoid '????' artifacts in OpenCV text."""
    if not text:
        return ""
    # Remove emojis and special symbols outside printable ASCII
    clean = re.sub(r'[^\x20-\x7E]', '', str(text)).strip()
    return clean if clean else "Face Recognition"


def _non_max_suppression(boxes, overlap_thresh=0.25):
    """Cleanly merge and eliminate overlapping bounding boxes on the same face."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)[::-1]
    pick = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        # Suppression based on overlap with either box (catches hands or sub-boxes inside head)
        overlap = intersection / np.minimum(area[i], area[idxs[1:]])

        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))

    return [tuple(int(v) for v in boxes[p]) for p in pick]


class FaceDetector:
    def __init__(self, method="auto", scale_factor=1.08, min_neighbors=4, min_size=(24, 24),
                 score_threshold=0.65, nms_threshold=0.25, crop_scale=CROP_SCALE):
        """
        method: 'auto', 'yunet', or 'haar'
        score_threshold: set to 0.65 to reject false positives (e.g. hands, books, backgrounds)
        """
        self.method = method
        self.crop_scale = crop_scale
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Initialize Haar Cascades
        self.frontal_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.profile_cascade = cv2.CascadeClassifier(HAAR_PROFILE_PATH) if os.path.exists(HAAR_PROFILE_PATH) else None

        # Initialize YuNet Deep Learning Detector
        self.yunet = None
        if hasattr(cv2, "FaceDetectorYN") and os.path.exists(YUNET_PATH):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    model=YUNET_PATH,
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=5000
                )
            except Exception as e:
                print(f"[FaceDetector] Warning: Could not initialize YuNet: {e}")
                self.yunet = None

    @staticmethod
    def _gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def _is_valid_face_geometry(self, face_landmarks, box):
        """
        Validate facial landmark geometry to strictly reject hands, neck, or random patterns:
        - Eyes must be horizontally spaced (> 8 px)
        - Mouth must be located strictly BELOW the eyes (y_mouth > y_eyes)
        - Aspect ratio (h/w) must match realistic human facial proportions (0.75 - 1.85)
        """
        bx, by, bw, bh = box
        aspect = bh / max(float(bw), 1.0)
        if aspect < 0.70 or aspect > 1.90:
            return False

        if len(face_landmarks) >= 14:
            # Landmarks: right_eye (4,5), left_eye (6,7), nose (8,9), right_mouth (10,11), left_mouth (12,13)
            re_x, re_y = face_landmarks[4], face_landmarks[5]
            le_x, le_y = face_landmarks[6], face_landmarks[7]
            eye_dist = np.hypot(re_x - le_x, re_y - le_y)
            if eye_dist < (bw * 0.15):  # Eyes must be reasonably separated
                return False

            eyes_y = (re_y + le_y) / 2.0
            mouth_y = (face_landmarks[11] + face_landmarks[13]) / 2.0
            if mouth_y <= eyes_y:  # Mouth must be below eyes
                return False

        return True

    def _detect_yunet(self, frame):
        """Ultra-fast deep learning detector with landmark verification."""
        if self.yunet is None:
            return self._detect_haar(frame)
        h, w = frame.shape[:2]
        self.yunet.setInputSize((w, h))
        _, faces = self.yunet.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        boxes = []
        for f in faces:
            score = float(f[-1])
            if score < self.score_threshold:
                continue

            bx, by, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(w - bx, bw), min(h - by, bh)

            if bw > 14 and bh > 14:
                box = (bx, by, bw, bh)
                if self._is_valid_face_geometry(f, box):
                    boxes.append(box)

        return _non_max_suppression(boxes, overlap_thresh=0.25)

    def _detect_haar(self, frame):
        """Viola-Jones detector with multi-angle fallback and Non-Maximum Suppression."""
        gray = cv2.equalizeHist(self._gray(frame))
        boxes = list(self.frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        ))

        if self.profile_cascade is not None:
            prof_boxes = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            if len(prof_boxes) > 0:
                boxes.extend(prof_boxes)

        boxes = [tuple(int(v) for v in b) for b in boxes]
        return _non_max_suppression(boxes, overlap_thresh=0.25)

    def detect(self, frame):
        """Detect all face bounding boxes [(x, y, w, h), ...] sorted by area."""
        if self.method == "yunet" and self.yunet is not None:
            boxes = self._detect_yunet(frame)
        elif self.method == "haar":
            boxes = self._detect_haar(frame)
        else:
            boxes = self._detect_yunet(frame) if self.yunet is not None else self._detect_haar(frame)

        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def crop(self, frame, box):
        """
        Crop face region with context padding (1.6x), resize to 112x92, apply CLAHE.
        Guarantees exact (112, 92, 1) float32 [0, 1] output required by the Siamese CNN.
        """
        gray = self._gray(frame)
        h_frame, w_frame = gray.shape[:2]
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0 - 0.04 * h
        cw = w * self.crop_scale
        ch = cw * (IMG_H / float(IMG_W))

        x0, y0 = int(round(cx - cw / 2.0)), int(round(cy - ch / 2.0))
        x1, y1 = int(round(x0 + cw)), int(round(y0 + ch))

        pad = int(max(cw, ch, 10))
        padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        crop_region = padded[y0 + pad:y1 + pad, x0 + pad:x1 + pad]

        if crop_region.size == 0 or crop_region.shape[0] < 5 or crop_region.shape[1] < 5:
            crop_region = gray[max(0, y):min(h_frame, y+h), max(0, x):min(w_frame, x+w)]

        return img_to_array(crop_region)

    def get_face(self, frame):
        boxes = self.detect(frame)
        if not boxes:
            return None
        return {"face": self.crop(frame, boxes[0]), "box": boxes[0], "num_faces": len(boxes)}

    def get_all_faces(self, frame):
        boxes = self.detect(frame)
        return [{"face": self.crop(frame, b), "box": b} for b in boxes]


def draw_boxes(frame, boxes, labels=None, colors=None, banner=None):
    """
    Draw high-visibility bounding boxes with solid background badge tags.
    Cleans emojis so text renders without '????' artifacts.
    """
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    s = max(out.shape[1] / 720.0, 0.5)

    for i, (x, y, w, h) in enumerate(boxes):
        col = colors[i] if colors else (0, 220, 0)
        thickness = max(int(2.5 * s), 2)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, thickness)

        if labels and i < len(labels):
            label_text = _strip_emojis(labels[i])
            font_scale = 0.55 * s
            font_thick = max(int(1.5 * s), 1)
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

            tag_y0 = max(y - text_h - 10, 0)
            tag_y1 = y
            tag_x0 = x
            tag_x1 = min(x + text_w + 10, out.shape[1])
            cv2.rectangle(out, (tag_x0, tag_y0), (tag_x1, tag_y1), col, -1)

            text_col = (255, 255, 255) if (col[0]*0.299 + col[1]*0.587 + col[2]*0.114) < 140 else (0, 0, 0)
            cv2.putText(out, label_text, (tag_x0 + 5, tag_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, font_thick, cv2.LINE_AA)

    if banner:
        raw_text, col = banner
        text = _strip_emojis(raw_text)
        banner_h = int(38 * s)
        cv2.rectangle(out, (0, 0), (out.shape[1], banner_h), (20, 24, 32), -1)
        cv2.putText(out, text, (12, int(26 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, col, max(int(2 * s), 1), cv2.LINE_AA)

    return out


# ---------------------------------------------------------------- Google Colab webcam helpers
_JS = '''
async function captureFrames(n, interval, quality, label) {
  const div = document.createElement('div');
  const btn = document.createElement('button');
  btn.textContent = label;
  btn.style.cssText = 'font-size:16px;padding:8px 16px;margin-bottom:8px';
  const status = document.createElement('div');
  status.style.cssText = 'font-family:sans-serif;margin:6px 0';
  const video = document.createElement('video');
  video.style.display = 'block';
  video.setAttribute('playsinline', '');
  div.appendChild(btn); div.appendChild(status); div.appendChild(video);
  document.body.appendChild(div);
  const stream = await navigator.mediaDevices.getUserMedia({video: true});
  video.srcObject = stream;
  await video.play();
  google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
  status.textContent = 'Look at the camera, then press the button.';
  await new Promise((resolve) => btn.onclick = resolve);
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  const frames = [];
  for (let i = 0; i < n; i++) {
    canvas.getContext('2d').drawImage(video, 0, 0);
    frames.push(canvas.toDataURL('image/jpeg', quality));
    status.textContent = 'Captured ' + (i + 1) + ' / ' + n + '  -  slightly turn your head / change expression';
    await new Promise((r) => setTimeout(r, interval));
  }
  stream.getVideoTracks()[0].stop();
  div.remove();
  return frames;
}
'''


def _decode(data_url):
    raw = base64.b64decode(data_url.split(",")[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def capture_frames_colab(n=1, interval_ms=700, quality=0.85, label="Start capture"):
    from IPython.display import display, Javascript
    from google.colab.output import eval_js
    display(Javascript(_JS))
    data = eval_js("captureFrames(%d, %d, %.2f, %s)" % (n, interval_ms, quality, json.dumps(label)))
    return [_decode(d) for d in data]


def frames_from_upload():
    from google.colab import files
    up = files.upload()
    frames = []
    for name, data in up.items():
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames
