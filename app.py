"""
app.py - Modern Web Application for Siamese Neural Network Face Recognition System
Lab Assignment 06 | CSE 4192: Machine Learning Projects with Python
Enhanced with Multi-Face Group Photo Recognition & Dual-Engine Detection
"""

import os
import sys
import time
import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Insert current directory into path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import database as db
from utils import (
    DB_PATH, MODEL_PATH, CONFIG_PATH, IMG_H, IMG_W, EMB_DIM, MARGIN,
    load_config, save_config, img_to_array
)
from face_detection import FaceDetector, draw_boxes
from face_recognition_system import FaceRecognizer
from onboarding import register_user

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepFace SNN | Biometric Recognition & Group Auth",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling & Glassmorphic CSS Theme
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }

    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
        background: rgba(99, 102, 241, 0.25);
        color: #c7d2fe;
        border: 1px solid rgba(165, 180, 252, 0.3);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }

    .status-granted {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid #10b981;
        color: #10b981;
        padding: 0.9rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 0.7rem 0;
    }

    .status-denied {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid #ef4444;
        color: #ef4444;
        padding: 0.9rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 0.7rem 0;
    }

    .status-group {
        background: rgba(14, 165, 233, 0.14);
        border: 1px solid #0ea5e9;
        color: #38bdf8;
        padding: 0.9rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 0.7rem 0;
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.14);
        border: 1px solid #f59e0b;
        color: #f59e0b;
        padding: 0.9rem 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 0.7rem 0;
    }

    .flow-step {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# System Loader & Helper Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    """Load the trained FaceRecognizer singleton."""
    return FaceRecognizer(matching_strategy="min")


def to_bgr(file):
    """Convert an uploaded file or camera frame to OpenCV BGR format."""
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)


# Initialize system
rec = load_system()
rec.refresh_users()
current_users = db.get_all_users()


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Meta
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.6rem 0 1rem 0;">
        <span style="font-size: 2.6rem;">🛡️</span>
        <h2 style="margin: 0.2rem 0 0 0; font-size: 1.3rem; font-weight: 700;">DeepFace SNN</h2>
        <p style="margin: 0; font-size: 0.8rem; color: #94a3b8;">Biometric Face & Group Auth</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation Menu",
        [
            "🏠 System Dashboard",
            "👤 Enroll New Identity",
            "🔍 Live Verification & Group Auth",
            "📊 Performance Analytics",
            "👥 Identity Directory",
            "🧪 Functional Test Suite",
            "📘 Architecture & Docs"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Advanced Engine Tuning Controls in Sidebar
    st.markdown("#### ⚡ Detection & Matching Engine")
    
    detector_choice = st.selectbox(
        "Face Detector Engine",
        ["⚡ YuNet Deep Learning (Best for Groups)", "🏛️ Viola–Jones Haar Cascade (Lab Std)"],
        index=0,
        help="YuNet detects tiny/rotated faces in crowded groups. Haar cascade is the classical baseline."
    )
    if "YuNet" in detector_choice:
        rec.detector.method = "yunet"
    else:
        rec.detector.method = "haar"

    strategy_choice = st.selectbox(
        "Gallery Matching Strategy",
        ["🎯 Best Reference Match (min distance)", "⚖️ Adaptive (70% Best + 30% Avg)", "📊 Top-3 Average (k=3)"],
        index=0,
        help="Best Reference Match compares the face against each enrolled pose and selects the closest match."
    )
    if "Best" in strategy_choice:
        rec.matching_strategy = "min"
    elif "Adaptive" in strategy_choice:
        rec.matching_strategy = "adaptive"
    else:
        rec.matching_strategy = "top_k"

    st.markdown("---")
    st.markdown(f"""
    - **Active Detector:** `{rec.detector.method.upper()}`
    - **Matching:** `{rec.matching_strategy}`
    - **Enrolled Users:** `{len(current_users)}`
    - **Threshold T:** `{rec.threshold:.4f}`
    """)
    st.caption("CSE 4192 | Lab Assignment 06\nITER, Siksha 'O' Anusandhan")


# =============================================================================
# 1. SYSTEM DASHBOARD (HOME)
# =============================================================================
if page == "🏠 System Dashboard":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Siamese Neural Network Biometric Verification</div>
        <div class="hero-subtitle">
            An end-to-end face recognition system capable of individual verification and crowd/group-picture identification.
            Powered by YuNet deep learning & Viola–Jones face detection, CLAHE image normalization, and a shared CNN embedding backbone.
        </div>
        <div>
            <span class="badge-pill">Deep Learning YuNet</span>
            <span class="badge-pill">Group Photo Recognition</span>
            <span class="badge-pill">Multi-Pose Gallery Matching</span>
            <span class="badge-pill">Contrastive Metric Loss</span>
            <span class="badge-pill">SQLite Embedding Store</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Indicators
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Enrolled Users", len(current_users), help="Total identities currently stored in SQLite database")
    
    total_embeddings = 0
    if os.path.exists(DB_PATH):
        import sqlite3
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT count(*) FROM embeddings").fetchone()
            total_embeddings = row[0] if row else 0
    kpi2.metric("Stored Embeddings", total_embeddings, help="Total 128-d reference embedding vectors across all poses")
    kpi3.metric("Test ROC-AUC", "0.786", delta="+0.286 vs chance", help="Area under the ROC curve on unseen identities")
    kpi4.metric("Test Accuracy", "72.04%", help="Pair accuracy evaluated at optimal threshold on unseen people")
    kpi5.metric("Decision Margin", f"{MARGIN:.1f}", help="Contrastive loss margin separating distinct identities")

    st.markdown("---")

    # Workflow Section
    st.subheader("🔄 End-to-End System Workflow")
    st.markdown("Every camera frame or group photo travels through a six-stage pipeline:")

    col_w1, col_w2, col_w3, col_w4, col_w5, col_w6 = st.columns(6)
    with col_w1:
        st.markdown("""
        <div class="flow-step">
            <strong>1. Input Image</strong><br>
            <span style="color:#94a3b8; font-size:0.75rem;">Single or Group Photo</span>
        </div>
        """, unsafe_allow_html=True)
    with col_w2:
        st.markdown("""
        <div class="flow-step">
            <strong>2. Dual Detector</strong><br>
            <span style="color:#94a3b8; font-size:0.75rem;">YuNet / Haar Multi-Face</span>
        </div>
        """, unsafe_allow_html=True)
    with col_w3:
        st.markdown("""
        <div class="flow-step">
            <strong>3. Preprocessing</strong><br>
            <span style="color:#94a3b8; font-size:0.75rem;">CLAHE + 112×92 Crop</span>
        </div>
        """, unsafe_allow_html=True)
    with col_w4:
        st.markdown("""
        <div class="flow-step">
            <strong>4. Twin CNN</strong><br>
            <span style="color:#94a3b8; font-size:0.75rem;">128-D Dense Embedding</span>
        </div>
        """, unsafe_allow_html=True)
    with col_w5:
        st.markdown("""
        <div class="flow-step">
            <strong>5. Multi-Pose Match</strong><br>
            <span style="color:#94a3b8; font-size:0.75rem;">Min Distance to Gallery</span>
        </div>
        """, unsafe_allow_html=True)
    with col_w6:
        st.markdown("""
        <div class="flow-step">
            <strong>6. Output Tagging</strong><br>
            <span style="color:#94a3b8; font-size:0.75rem;">Labeled Bounding Boxes</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("""
        <div class="glass-card">
            <h4>💡 Breakthrough: Group Photo Identification</h4>
            <p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                Identifying multiple individuals in a group photo poses unique challenges:
                faces are smaller ($20\text{--}40$ px), people look in different directions, and expressions vary.
            </p>
            <p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
                We solved this through two major enhancements:
                <br>1. <strong>YuNet Deep Learning Detector:</strong> Locates faces of any scale down to $10 \times 10$ pixels at any angle.
                <br>2. <strong>Multi-Pose Gallery Matching:</strong> Compares each face against all enrolled poses (neutral, smiling, tilted) using minimum Euclidean distance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div class="glass-card">
            <h4>🚀 Quick Action Center</h4>
            <p style="font-size: 0.9rem; color: #cbd5e1;">Explore the core features:</p>
            <ul style="font-size: 0.9rem; color: #94a3b8; line-height: 1.7;">
                <li><strong>Enroll New User:</strong> Capture 5-10 facial photos (neutral, smiling, left, right) to build a rich reference gallery.</li>
                <li><strong>Group Photo Auth:</strong> Upload a group picture with friends or classmates to identify everyone simultaneously.</li>
                <li><strong>Threshold Presets:</strong> Switch between High Security ($T=0.50$), Standard ($T=0.6506$), and High Convenience ($T=0.80$).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 2. ENROLL NEW IDENTITY (ONBOARDING)
# =============================================================================
elif page == "👤 Enroll New Identity":
    st.header("👤 Identity Onboarding & Enrolment")
    st.markdown("Register a new user identity or add additional reference poses to an existing identity. Multi-pose enrollment significantly boosts group photo recognition!")

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        new_uid = st.text_input("User ID (e.g. U003, EMP4192, SOA2026)", placeholder="Enter unique ID...")
    with col_meta2:
        new_name = st.text_input("Full Name", placeholder="Enter user's name...")

    if "reg_frames" not in st.session_state:
        st.session_state.reg_frames = []
        st.session_state.cam_counter = 0

    tab_cam, tab_upload = st.tabs(["📸 Webcam Capture (Interactive)", "📁 File Batch Upload"])

    with tab_cam:
        st.markdown("##### Capture Face Samples")
        st.caption("Capture 5 to 10 photos with slight variations in head angle (frontal, slight left, slight right, smiling) for best accuracy in group photos.")

        cam_pic = st.camera_input(
            "Look at the camera and snap a photo",
            key=f"cam_enroll_{st.session_state.cam_counter}"
        )

        col_b1, col_b2 = st.columns([1, 4])
        with col_b1:
            if cam_pic is not None:
                if st.button("➕ Add This Sample", type="secondary"):
                    st.session_state.reg_frames.append(to_bgr(cam_pic))
                    st.session_state.cam_counter += 1
                    st.rerun()
        with col_b2:
            if st.session_state.reg_frames:
                if st.button("🧹 Clear Captured Samples"):
                    st.session_state.reg_frames = []
                    st.rerun()

    with tab_upload:
        st.markdown("##### Batch File Upload")
        st.caption("Upload multiple individual photos or crops of the person.")
        uploaded_files = st.file_uploader(
            "Select face images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

    uploaded_frames = [to_bgr(u) for u in (uploaded_files or [])]
    total_frames = st.session_state.reg_frames + uploaded_frames
    
    st.markdown("---")
    st.markdown(f"**Total Samples Prepared:** `{len(total_frames)}` (Camera: `{len(st.session_state.reg_frames)}`, Uploaded: `{len(uploaded_frames)}`)")

    progress_val = min(1.0, len(total_frames) / 5.0)
    st.progress(progress_val, text=f"Collected {len(total_frames)} / 5 recommended samples")

    overwrite = st.checkbox("Overwrite/Update user if ID already exists", value=True)

    if st.button("🚀 Register & Store Identity", type="primary", disabled=(len(total_frames) < 3)):
        if not new_uid or not new_name:
            st.error("Please provide both a User ID and Full Name before registering.")
        else:
            with st.spinner("Detecting faces, generating 128-D embeddings, and committing to SQLite..."):
                res = register_user(
                    user_id=new_uid,
                    name=new_name,
                    frames=total_frames,
                    embedder=rec.embedder,
                    detector=rec.detector,
                    min_faces=3,
                    overwrite=overwrite,
                    verbose=False
                )

            if res["success"]:
                st.balloons()
                st.markdown(f"""
                <div class="status-granted">
                    {res['message']}
                </div>
                """, unsafe_allow_html=True)
                st.session_state.reg_frames = []
                rec.refresh_users()

                if "crops" in res and res["crops"]:
                    st.markdown("##### 🔍 Stored & Preprocessed Face Templates (112×92)")
                    crop_cols = st.columns(min(len(res["crops"]), 8))
                    for idx, c in enumerate(res["crops"][:8]):
                        crop_cols[idx].image(c, caption=f"Template #{idx+1}", width=90)
            else:
                st.markdown(f"""
                <div class="status-denied">
                    ❌ {res['message']}
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# 3. LIVE VERIFICATION & GROUP AUTHENTICATION
# =============================================================================
elif page == "🔍 Live Verification & Group Auth":
    st.header("🔍 Real-Time Biometric Verification & Group Photo Auth")
    st.markdown("Test single-person verification or **upload a group picture** with multiple people to identify registered users.")

    st.markdown("##### ⚙️ Decision Sensitivity & Security Policy")
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("🔒 High Security (T = 0.50)", help="Strictest rejection of unknown users"):
        rec.threshold = 0.50
    if col_p2.button("⚖️ Standard Baseline (T = 0.6506)", help="Assignment 6 optimal accuracy point"):
        rec.threshold = 0.6506
    if col_p3.button("🚀 High Convenience (T = 0.80)", help="Minimizes false rejection for group photos"):
        rec.threshold = 0.80

    rec.threshold = st.slider(
        "Active Decision Threshold T (Euclidean Distance)",
        min_value=0.10,
        max_value=2.00,
        value=float(rec.threshold),
        step=0.01,
        help="Embeddings with distance below this value are recognized as registered users."
    )

    st.markdown("---")

    input_mode = st.radio("Choose Input Source", ["📁 Upload Image / Group Photo", "📸 Live Webcam Snapshot"], horizontal=True)
    probe_frame = None

    if input_mode == "📸 Live Webcam Snapshot":
        cam_in = st.camera_input("Look into the camera")
        if cam_in is not None:
            probe_frame = to_bgr(cam_in)
    else:
        file_in = st.file_uploader("Upload an image (single portrait or group picture)", type=["jpg", "jpeg", "png"])
        if file_in is not None:
            probe_frame = to_bgr(file_in)

    if probe_frame is not None:
        with st.spinner(f"Analyzing frame using {rec.detector.method.upper()} detector and Siamese CNN..."):
            res = rec.recognize_frame(probe_frame)

        annotated = rec.annotate(res)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Main Banner Output
        if res["status"] == "MULTIPLE_FACES":
            st.markdown(f"""
            <div class="status-group">
                {res['message']}
            </div>
            """, unsafe_allow_html=True)
        elif res["status"] == "RECOGNIZED":
            st.markdown(f"""
            <div class="status-granted">
                ✅ {res['message']}
            </div>
            """, unsafe_allow_html=True)
        elif res["status"] == "UNKNOWN":
            st.markdown(f"""
            <div class="status-denied">
                🚫 {res['message']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-warning">
                ⚠️ {res['message']}
            </div>
            """, unsafe_allow_html=True)

        col_img, col_metrics = st.columns([1.3, 1.0])

        with col_img:
            st.markdown("##### 📸 Detection & Tagging Canvas")
            st.image(annotated_rgb, use_container_width=True)

        with col_metrics:
            st.markdown("##### 📊 Recognition Summary")
            st.metric("Total Faces Detected", res["n_faces"])
            st.metric("Enrolled Members Identified", res["n_recognized"], delta=f"{res['n_recognized']}/{max(res['n_faces'],1)} match")
            st.metric("Active Decision Threshold", f"{rec.threshold:.4f}")
            st.caption(f"Detector: **{rec.detector.method.upper()}** | Strategy: **{rec.matching_strategy}**")

        # Detailed Group Photo Inspector
        if res["faces"]:
            st.markdown("---")
            st.subheader("👥 Individual Face Breakdown")
            st.caption("Inspect each person identified in the photo along with their measured distance to the registered database:")

            for i, face_res in enumerate(res["faces"]):
                with st.expander(f"👤 Face #{i+1}: {face_res['name']} ({'✅ MATCH' if face_res['recognized'] else '❌ GUEST'})", expanded=(i < 3)):
                    c_face1, c_face2, c_face3 = st.columns([1, 1.2, 1.8])

                    with c_face1:
                        st.markdown("**Face in Photo:**")
                        face_disp = (face_res["face_crop"][..., 0] * 255).astype(np.uint8)
                        st.image(face_disp, width=100, caption=f"Box: {face_res['box']}")

                    with c_face2:
                        st.markdown("**Closest Reference:**")
                        closest_uid = face_res.get("closest_uid")
                        best_sample_idx = face_res.get("best_sample_idx", 0)
                        ref_folder = os.path.join(os.path.dirname(DB_PATH), str(closest_uid)) if closest_uid else None
                        
                        ref_shown = False
                        if ref_folder and os.path.isdir(ref_folder):
                            ref_images = sorted([os.path.join(ref_folder, f) for f in os.listdir(ref_folder) if f.endswith(".png")])
                            if ref_images and best_sample_idx < len(ref_images):
                                st.image(ref_images[best_sample_idx], width=100, caption=f"{face_res['closest_user']} (Ref #{best_sample_idx+1})")
                                ref_shown = True
                            elif ref_images:
                                st.image(ref_images[0], width=100, caption=f"{face_res['closest_user']}")
                                ref_shown = True
                        if not ref_shown:
                            st.info("No photo on disk")

                    with c_face3:
                        st.markdown("**Biometric Verification Data:**")
                        st.write(f"- **Decision:** `{'ACCESS GRANTED' if face_res['recognized'] else 'ACCESS DENIED'}`")
                        st.write(f"- **Closest User:** `{face_res['closest_user']}`")
                        st.write(f"- **Distance (d):** `{face_res['distance']:.4f}` (Threshold: `{rec.threshold:.4f}`)")
                        st.write(f"- **Confidence:** `{face_res['confidence']}`")
                        st.progress(face_res['sim_score'] / 100.0, text=f"Match Score: {face_res['sim_score']}%")

                    # Distance breakdown to all registered users
                    st.markdown("**Distance to All Registered Users:**")
                    dist_df = pd.DataFrame(
                        list(face_res["distances"].items()),
                        columns=["User Name", "Measured Distance"]
                    ).sort_values(by="Measured Distance")
                    dist_df["Status"] = dist_df["Measured Distance"].apply(lambda d: "✅ Verified" if d < rec.threshold else "❌ Impostor")
                    st.dataframe(dist_df, hide_index=True, use_container_width=True)


# =============================================================================
# 4. PERFORMANCE ANALYTICS & METRICS
# =============================================================================
elif page == "📊 Performance Analytics":
    st.header("📊 Siamese Model Performance Analytics")
    st.markdown("Detailed quantitative evaluation of the Siamese neural network on held-out test identities (never seen during training).")

    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    col_m1.metric("Accuracy", "72.04%")
    col_m2.metric("ROC-AUC", "0.786")
    col_m3.metric("Precision", "70.17%")
    col_m4.metric("Recall", "76.67%")
    col_m5.metric("FAR", "32.59%", help="False Accept Rate (impostors wrongly accepted)")
    col_m6.metric("FRR", "23.33%", help="False Reject Rate (genuine users rejected)")

    tab_eval, tab_curves, tab_pca, tab_data = st.tabs([
        "📈 ROC, Confusion Matrix & Thresholds",
        "📉 28-Epoch Training History",
        "🌌 Embedding PCA Projection",
        "📋 Full Metric Table"
    ])

    with tab_eval:
        st.markdown("##### Comprehensive Test Set Evaluation")
        st.markdown("Evaluated on **540 test pairs** constructed from 6 unseen individuals (270 positive, 270 negative).")
        eval_img_path = os.path.join(BASE_DIR, "outputs", "07_evaluation.png")
        if os.path.exists(eval_img_path):
            st.image(eval_img_path, caption="Evaluation Plots: Distance Distribution, ROC Curve, Confusion Matrix, and Threshold Trade-off", use_container_width=True)

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.markdown("##### 🧮 Confusion Matrix Breakdown (T = 0.6506)")
            cm_data = pd.DataFrame(
                [[182, 88], [63, 207]],
                columns=["Predicted Different", "Predicted Same"],
                index=["True Different", "True Same"]
            )
            st.dataframe(cm_data, use_container_width=True)
        with col_cm2:
            st.markdown("##### ⚖️ Security Trade-Off Analysis")
            st.markdown("""
            - **True Positives (TP):** 207 matching pairs correctly verified.
            - **True Negatives (TN):** 182 impostor pairs correctly blocked.
            - **False Positives (FP):** 88 impostor pairs wrongly accepted (FAR 32.59%).
            - **False Negatives (FN):** 63 genuine pairs wrongly rejected (FRR 23.33%).
            """)

    with tab_curves:
        st.markdown("##### Two-Phase Training Convergence")
        st.markdown("Phase A (10 fixed warm-up epochs) followed by Phase B (fine-tuning with Early Stopping, best weights restored from epoch 20).")
        curves_img_path = os.path.join(BASE_DIR, "outputs", "06_training_curves.png")
        if os.path.exists(curves_img_path):
            st.image(curves_img_path, caption="Loss and Accuracy Curves across 28 Training Epochs", use_container_width=True)

        history_csv = os.path.join(BASE_DIR, "outputs", "training_history.csv")
        if os.path.exists(history_csv):
            st.markdown("##### Epoch-by-Epoch Logged Training History")
            hist_df = pd.read_csv(history_csv)
            st.dataframe(hist_df, use_container_width=True)

    with tab_pca:
        st.markdown("##### 128-D Embedding Space (PCA 2D Projection)")
        st.markdown("PCA visualization of 128-dimensional facial embeddings for the six unseen test identities.")
        pca_img_path = os.path.join(BASE_DIR, "outputs", "08_embedding_pca.png")
        if os.path.exists(pca_img_path):
            st.image(pca_img_path, caption="2D PCA Clusters of Unseen Identities", use_container_width=True)

    with tab_data:
        st.markdown("##### Full Statistical Evaluation Metrics")
        stats_df = pd.DataFrame([
            {"Metric": "Accuracy", "Value": "0.7204", "Formula": "(TP + TN) / Total", "Interpretation": "Overall verification rate"},
            {"Metric": "ROC-AUC", "Value": "0.786", "Formula": "Area under ROC", "Interpretation": "Threshold-independent discriminability"},
            {"Metric": "Precision", "Value": "0.7017", "Formula": "TP / (TP + FP)", "Interpretation": "Confidence in an accepted match"},
            {"Metric": "Recall (Sensitivity)", "Value": "0.7667", "Formula": "TP / (TP + FN)", "Interpretation": "Ability to recognize genuine users"},
            {"Metric": "Specificity", "Value": "0.6741", "Formula": "TN / (TN + FP)", "Interpretation": "Ability to reject impostors"},
            {"Metric": "F1-Score", "Value": "0.7327", "Formula": "2 * (P * R) / (P + R)", "Interpretation": "Harmonic balance of precision and recall"},
            {"Metric": "False Accept Rate (FAR)", "Value": "0.3259", "Formula": "FP / (FP + TN)", "Interpretation": "Rate at which impostors are admitted"},
            {"Metric": "False Reject Rate (FRR)", "Value": "0.2333", "Formula": "FN / (FN + TP)", "Interpretation": "Rate at which genuine users are blocked"},
        ])
        st.dataframe(stats_df, hide_index=True, use_container_width=True)


# =============================================================================
# 5. IDENTITY DIRECTORY & MANAGEMENT
# =============================================================================
elif page == "👥 Identity Directory":
    st.header("👥 Enrolled Identity Directory")
    st.markdown("Browse, inspect, and manage identities enrolled in the biometric SQLite database.")

    users = db.get_all_users()
    if not users:
        st.info("No identities enrolled yet. Navigate to 'Enroll New Identity' to register your first user.")
    else:
        st.markdown(f"**Total Registered Users:** `{len(users)}`")
        users_df = pd.DataFrame(users)
        st.dataframe(users_df, use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Inspect Stored Reference Faces")

        selected_user = st.selectbox(
            "Select an enrolled user to view reference samples:",
            [f"{u['user_id']} - {u['name']}" for u in users]
        )

        sel_uid = selected_user.split(" - ")[0]
        user_folder = os.path.join(os.path.dirname(DB_PATH), sel_uid)

        if os.path.isdir(user_folder):
            crops = sorted([os.path.join(user_folder, f) for f in os.listdir(user_folder) if f.endswith(".png")])
            if crops:
                st.write(f"Displaying **{len(crops)}** reference face samples stored for `{selected_user}`:")
                cols = st.columns(min(len(crops), 6))
                for i, c_path in enumerate(crops[:12]):
                    cols[i % 6].image(c_path, caption=f"Sample #{i+1}", width=90)
            else:
                st.warning("No image crops found on disk for this user.")

        st.markdown("---")
        st.subheader("🗑️ Identity Management")
        del_col1, del_col2 = st.columns([2, 1])
        with del_col1:
            user_to_delete = st.selectbox("Select user to remove permanently:", [f"{u['user_id']} - {u['name']}" for u in users], key="del_select")
        with del_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Delete Identity", type="primary"):
                del_uid = user_to_delete.split(" - ")[0]
                db.delete_user(del_uid)
                rec.refresh_users()
                st.success(f"Successfully deleted {user_to_delete} from the database.")
                st.rerun()


# =============================================================================
# 6. FUNCTIONAL TEST SUITE (TC01-TC12)
# =============================================================================
elif page == "🧪 Functional Test Suite":
    st.header("🧪 Functional Test Suite (TC01 – TC12)")
    st.markdown("Verification of all 12 mandatory system integration test cases specified in Lab Assignment 06.")

    test_file = os.path.join(BASE_DIR, "test_cases", "test_results.csv")
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)

        n_pass = (test_df["Status"] == "Pass").sum()
        n_fail = (test_df["Status"] == "Fail").sum()

        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("Total Test Cases", len(test_df))
        t_col2.metric("Tests Passed", n_pass, delta=f"{n_pass/len(test_df)*100:.1f}%")
        t_col3.metric("Tests Failed", n_fail, delta="-1", delta_color="inverse")

        st.markdown("---")
        st.markdown("##### Detailed Test Results Table")
        st.dataframe(test_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔍 Deep Dive: The Documented TC10 Failure")
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #f87171;">⚠️ Why did TC10 fail (67% rejection)?</h4>
            <p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.6;">
                <strong>Test Case 10</strong> evaluates the rejection of unknown (unregistered) impostors appearing in front of the camera.
                During live testing, 20 out of 30 unknown face frames were correctly rejected as <code>UNKNOWN</code>, while 10 frames were mistakenly accepted.
            </p>
            <p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.6;">
                This <strong>67% live rejection rate</strong> directly mirrors the measured offline <strong>False Accept Rate (FAR) of 32.59%</strong> at <code>T = 0.6506</code>.
                With our enhanced <strong>YuNet detector + Best Pose Matching</strong>, rejection and recognition boundaries are significantly sharpened.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Test results file `test_cases/test_results.csv` not found.")


# =============================================================================
# 7. ARCHITECTURE & DOCUMENTATION
# =============================================================================
elif page == "📘 Architecture & Docs":
    st.header("📘 System Architecture & Technical Specifications")

    st.markdown("##### 🧬 Siamese Neural Network Architecture")
    arch_img_path = os.path.join(BASE_DIR, "outputs", "04_siamese_architecture.png")
    if os.path.exists(arch_img_path):
        st.image(arch_img_path, caption="Siamese Network Architecture Diagram", use_container_width=True)

    st.markdown("##### 📐 Mathematical Formulations")
    st.markdown("**1. Contrastive Loss Function (Hadsell et al., 2006):**")
    st.latex(r"\mathcal{L}(y, d) = y \cdot d^2 + (1 - y) \cdot \max(m - d, 0)^2")
    st.markdown(r"""
    Where:
    - $d = \|f(x_1) - f(x_2)\|_2$ is the Euclidean distance between embeddings.
    - $y = 1$ for genuine (same-person) pairs; $y = 0$ for impostor (different-person) pairs.
    - $m = 1.0$ is the contrastive margin.
    """)

    st.markdown("##### 🧱 Detailed Layer Configuration")
    layer_df = pd.DataFrame([
        {"Layer": "Input", "Output Shape": "(112, 92, 1)", "Parameters": "0", "Details": "Grayscale face image"},
        {"Layer": "Conv2D", "Output Shape": "(110, 90, 64)", "Parameters": "640", "Details": "64 filters, 3×3, ReLU"},
        {"Layer": "MaxPooling2D", "Output Shape": "(55, 45, 64)", "Parameters": "0", "Details": "2×2 pool size"},
        {"Layer": "Conv2D", "Output Shape": "(53, 43, 64)", "Parameters": "36,928", "Details": "64 filters, 3×3, ReLU"},
        {"Layer": "Flatten", "Output Shape": "(145,856)", "Parameters": "0", "Details": "Feature map serialization"},
        {"Layer": "Dense (Embedding)", "Output Shape": "(128)", "Parameters": "18,669,696", "Details": "128 units, Sigmoid activation"},
        {"Layer": "Total Trainable", "Output Shape": "128-D", "Parameters": "18,707,264", "Details": "Dominated by Flatten to Dense projection"}
    ])
    st.dataframe(layer_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("👥 Project Attribution & Laboratory Record")
    st.markdown("""
    - **Course:** Machine Learning Projects with Python (CSE 4192)
    - **Institution:** Centre for Artificial Intelligence & Machine Learning, Faculty of Engineering & Technology (ITER), Siksha 'O' Anusandhan, Bhubaneswar, Odisha
    - **Authors:** Rishav Raj, Saptorshi Kumar Paul, Anubhab Samantaray, Suprava Pattanayak
    - **Dataset Acknowledgement:** AT&T Laboratories Cambridge (ORL Database of Faces)
    """)
