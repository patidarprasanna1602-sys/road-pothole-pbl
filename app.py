
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 1. Prevent Ultralytics permission errors on Windows AppData
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))

# 2. Automatically link .venv site-packages if running from an unactivated terminal
VENV_SITE = BASE_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))

import streamlit as st
import streamlit.runtime

# If executed directly with 'python app.py', automatically launch streamlit
if not streamlit.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())

from PIL import Image
from io import BytesIO
from collections import Counter
import pandas as pd

from src.predictor import predict_image, predict_image_sahi, CLASS_DESCRIPTIONS
from src.video_processor import process_video


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🚧 AI-Based Urban Road Damage Detection")

st.markdown(
    """
    An intelligent road damage detection system using
    **YOLO11n**, trained on the combined **RDD2020 and RDD2022** datasets
    with **10 damage categories** including Potholes, Cracks, and Surface Distress.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

ALL_CLASSES = list(CLASS_DESCRIPTIONS.keys())

st.sidebar.title("🛠️ Detection Controls")

# --- Inference Mode ---
use_sahi = st.sidebar.toggle(
    "🔬 High-Accuracy Mode (SAHI Sliced Inference)",
    value=False,
    help=(
        "SAHI slices the image into overlapping tiles and runs YOLO on each tile. "
        "Turn OFF for large close-up potholes (standard mode is better). "
        "Turn ON for small or distant damage."
    )
)

st.sidebar.divider()

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=0.95,
    value=0.15,
    step=0.01,
    help="Lower = more detections. Keep at 0.10–0.20 for best results."
)

# Fixed NMS IoU threshold (not exposed to user)
iou_threshold = 0.45

if use_sahi:
    slice_size = st.sidebar.selectbox("Slice Size (px)", [512, 640, 416, 320, 256], index=0,
        help="Use 512–640 for large potholes. Use 256 for small distant ones.")
    overlap_ratio = 0.3
else:
    agnostic_nms = True
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        imgsz = st.selectbox("Image Size", [640, 1280, 960], index=0,
            help="640 is standard. Use 1280 for very high-res images.")
    with col_s2:
        augment = st.checkbox("Enable TTA", value=False)

selected_classes = st.sidebar.multiselect(
    "Filter Damage Classes",
    options=ALL_CLASSES,
    default=ALL_CLASSES,
    format_func=lambda x: f"{x}: {CLASS_DESCRIPTIONS.get(x, x)}",
    help="Select which damage types to detect. If EMPTY, all classes are detected."
)

# If user clears all selections, detect all classes
if not selected_classes:
    selected_classes = ALL_CLASSES

st.sidebar.button("🎯 Show Only Potholes (D20 + D40)", use_container_width=True,
    help="Click then re-detect to filter only pothole/rutting damage.")

st.sidebar.divider()

st.sidebar.info(
    """
    💡 **Damage Class Guide:**
    * **D20**: True Pothole (cavity/hole)
    * **D40**: Rutting / Surface Depression (water-filled)
    * **D00 / D01**: Longitudinal / Lateral Cracks
    * **D10 / D11**: Alligator / Complex Cracks
    * **D43 / D44**: Faded Markings
    * **D50**: General Road Damage
    """
)


# --------------------------------------------------
# MAIN WORKSPACE: IMAGE & VIDEO MODES
# --------------------------------------------------

tab_image, tab_video = st.tabs([
    "📸 Single Image Inspection",
    "🎥 Video Inspection & Pothole Frame Capture"
])


# ==================================================
# TAB 1: SINGLE IMAGE INSPECTION
# ==================================================

with tab_image:
    st.subheader("Upload Road Image")

    uploaded_file = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.caption(
                f"Image dimensions: {image.width} × {image.height}"
            )

            st.divider()

            # --------------------------------------------------
            # ORIGINAL IMAGE
            # --------------------------------------------------

            left, right = st.columns(2)

            with left:
                st.subheader("Original Image")
                st.image(
                    image,
                    use_container_width=True
                )

            # --------------------------------------------------
            # DETECTION
            # --------------------------------------------------

            if st.button(
                "🔍 Detect Road Damage",
                type="primary",
                use_container_width=True
            ):

                with st.spinner("Analyzing road image..."):

                    try:
                        # Always pass None filter when all classes are selected (safest)
                        cls_filter = selected_classes if len(selected_classes) < len(ALL_CLASSES) else None

                        if use_sahi:
                            annotated_image, detections = predict_image_sahi(
                                image,
                                confidence=confidence,
                                iou_threshold=iou_threshold,
                                slice_size=slice_size,
                                overlap_ratio=overlap_ratio,
                                filter_classes=cls_filter,
                            )
                        else:
                            annotated_image, detections = predict_image(
                                image,
                                confidence=confidence,
                                iou_threshold=iou_threshold,
                                agnostic_nms=agnostic_nms,
                                imgsz=imgsz,
                                augment=augment,
                                filter_classes=cls_filter,
                            )

                        # annotated_image is already RGB from predictor
                        result_image = Image.fromarray(annotated_image)

                        with right:
                            st.subheader("Detection Results")

                            st.image(
                                result_image,
                                caption="Road Damage Detection Result",
                                use_container_width=True
                            )

                        st.success("Detection completed successfully!")

                        st.divider()

                        # --------------------------------------------------
                        # DETECTION SUMMARY
                        # --------------------------------------------------

                        st.subheader("Detection Summary")

                        total_detections = len(detections)

                        class_counts = Counter(
                            item["class"] for item in detections
                        )

                        unique_classes = len(class_counts)

                        average_confidence = (
                            sum(item["confidence"] for item in detections)
                            / total_detections
                            if total_detections > 0
                            else 0
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Total Detections",
                            total_detections
                        )

                        col2.metric(
                            "Damage Classes Detected",
                            unique_classes
                        )

                        col3.metric(
                            "Average Confidence",
                            f"{average_confidence:.2%}"
                        )

                        st.caption(
                            f"Confidence threshold used: {confidence:.2f}"
                        )

                        # --------------------------------------------------
                        # CLASS-WISE ANALYSIS
                        # --------------------------------------------------

                        st.divider()
                        st.subheader("Class-wise Detection Analysis")

                        if detections:

                            class_df = pd.DataFrame(
                                [
                                    {
                                        "Damage Class": class_name,
                                        "Number of Detections": count
                                    }
                                    for class_name, count
                                    in sorted(class_counts.items())
                                ]
                            )

                            chart_col, table_col = st.columns(2)

                            with chart_col:
                                st.markdown("**Detection Count by Class**")

                                st.bar_chart(
                                    class_df.set_index("Damage Class")
                                )

                            with table_col:
                                st.markdown("**Class-wise Counts**")

                                st.dataframe(
                                    class_df,
                                    use_container_width=True,
                                    hide_index=True
                                )

                            # --------------------------------------------------
                            # DETECTION DETAILS
                            # --------------------------------------------------

                            st.divider()
                            st.subheader("Detected Damage Details")

                            detection_df = pd.DataFrame(detections)

                            rename_cols = {
                                "class": "Class Code",
                                "description": "Damage Type",
                                "severity": "Severity",
                                "confidence": "Confidence",
                                "x1": "X1 (Left)",
                                "y1": "Y1 (Top)",
                                "x2": "X2 (Right)",
                                "y2": "Y2 (Bottom)"
                            }
                            detection_df = detection_df.rename(columns=rename_cols)

                            st.dataframe(
                                detection_df,
                                use_container_width=True,
                                hide_index=True
                            )

                        else:
                            st.warning(
                                "No road damage detected at this "
                                "confidence threshold. Try lowering "
                                "the threshold."
                            )

                        # --------------------------------------------------
                        # DOWNLOAD RESULT
                        # --------------------------------------------------

                        st.divider()
                        st.subheader("Download Results")

                        output = BytesIO()
                        result_image.save(output, format="PNG")

                        st.download_button(
                            label="⬇️ Download Annotated Image",
                            data=output.getvalue(),
                            file_name="road_damage_result.png",
                            mime="image/png",
                            type="primary"
                        )

                    except Exception as error:
                        st.error(
                            f"Prediction failed: {error}"
                        )

        except Exception as error:
            st.error(
                f"Unable to process the uploaded image: {error}"
            )


# ==================================================
# TAB 2: VIDEO INSPECTION & POTHOLE CAPTURE
# ==================================================

with tab_video:
    st.subheader("🎥 Dashcam / Road Video Inspection")
    st.markdown(
        """
        Upload a road inspection video to scan for road hazards frame-by-frame.
        The system will **automatically capture and log specific frames where potholes or damage occur**,
        complete with timestamps, severity ratings, and individual frame downloads.
        """
    )

    video_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
        help="Upload dashcam footage (.mp4, .avi, .mov, or .mkv)."
    )

    if video_file is not None:
        col_v1, col_v2 = st.columns([1, 1])

        with col_v1:
            st.markdown("**Original Video Preview**")
            st.video(video_file)

        with col_v2:
            st.markdown("**⚙️ Video Processing Controls**")
            v_frame_skip = st.slider(
                "Frame Sampling (Process every N frames)",
                min_value=1,
                max_value=10,
                value=2,
                help="1 = Every frame (most thorough, slower), 2 = Every 2nd frame (2x faster, recommended), 5 = Quick preview."
            )
            v_pothole_only = st.checkbox(
                "🎯 Capture Only Potholes & Rutting (D20 & D40)",
                value=True,
                help="When checked, only logs frames containing potholes or rutting depressions, ignoring pure lane markings or cracks."
            )
            v_max_seconds = st.number_input(
                "Maximum video duration to scan (seconds, 0 = entire video)",
                min_value=0,
                max_value=600,
                value=30,
                step=5,
                help="Limits video processing time for quick demonstration."
            )

        start_video_btn = st.button(
            "🚀 Scan Video & Capture Pothole Frames",
            type="primary",
            use_container_width=True
        )

        if start_video_btn:
            temp_dir = BASE_DIR / "outputs" / "video_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_input = temp_dir / "input_uploaded_video.mp4"
            temp_output = temp_dir / "annotated_result_video.mp4"

            # Save uploaded video bytes to disk
            with open(temp_input, "wb") as f:
                f.write(video_file.getbuffer())

            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            def update_progress(current_frame, total_frames, incidents_count, fps):
                pct = min(current_frame / total_frames, 1.0) if total_frames > 0 else 0
                progress_bar.progress(pct)
                status_placeholder.markdown(
                    f"⏳ **Scanning Video:** `{current_frame}/{total_frames}` frames ({pct:.0%}) | "
                    f"**Speed:** `{fps} FPS` | "
                    f"**Pothole Frames Captured:** `{incidents_count}`"
                )

            cls_filter = selected_classes if len(selected_classes) < len(ALL_CLASSES) else None

            # Calculate max_frames from max_seconds
            max_f = None
            if v_max_seconds > 0:
                import cv2
                probe_cap = cv2.VideoCapture(str(temp_input))
                probe_fps = probe_cap.get(cv2.CAP_PROP_FPS) or 25.0
                probe_cap.release()
                max_f = int(v_max_seconds * probe_fps)

            with st.spinner("Processing video and extracting damage frames..."):
                try:
                    res = process_video(
                        video_path=temp_input,
                        output_path=temp_output,
                        confidence=confidence,
                        iou_threshold=iou_threshold,
                        frame_skip=v_frame_skip,
                        imgsz=imgsz if not use_sahi else 640,
                        filter_classes=cls_filter,
                        pothole_only=v_pothole_only,
                        max_frames=max_f,
                        progress_callback=update_progress,
                    )

                    st.success("✅ Video analysis complete!")

                    # --------------------------------------------------
                    # VIDEO SUMMARY METRICS
                    # --------------------------------------------------
                    st.divider()
                    st.subheader("📊 Video Analysis Summary")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Frames Analyzed", res["processed_frames"])
                    m2.metric("Incident Frames Captured", len(res["incident_frames"]))
                    m3.metric("Total Detections", res["total_detections"])
                    m4.metric("Processing Speed", f"{res['processing_fps']} FPS")

                    # --------------------------------------------------
                    # CAPTURED INCIDENT FRAMES GALLERY
                    # --------------------------------------------------
                    st.divider()
                    st.subheader("🎯 Captured Pothole & Road Damage Frames")

                    incidents = res["incident_frames"]
                    if incidents:
                        st.write(
                            f"Identified **{len(incidents)} key frames** with detected road damage:"
                        )

                        for i, inc in enumerate(incidents):
                            classes_str = ", ".join(inc["classes"])
                            header_title = (
                                f"📍 Frame #{inc['frame_idx']} — Timestamp: {inc['timestamp_str']} | "
                                f"Damage: {classes_str} | "
                                f"Severity: {inc['highest_severity']} (Conf: {inc['max_confidence']:.2f})"
                            )

                            with st.expander(header_title, expanded=(i < 3)):
                                c_img, c_meta = st.columns([2, 1])

                                frame_pil = Image.fromarray(inc["annotated_image"])
                                with c_img:
                                    st.image(
                                        frame_pil,
                                        caption=f"Incident Frame #{inc['frame_idx']} ({inc['timestamp_str']})",
                                        use_container_width=True
                                    )

                                with c_meta:
                                    st.markdown(f"**Timestamp:** `{inc['timestamp_str']}`")
                                    st.markdown(f"**Video Frame Index:** `#{inc['frame_idx']}`")
                                    st.markdown(f"**Highest Severity:** `{inc['highest_severity']}`")
                                    st.markdown(f"**Damage Count:** `{inc['detections_count']}`")
                                    st.markdown(f"**Max Confidence:** `{inc['max_confidence']:.2%}`")

                                    # Download this specific frame button
                                    buf = BytesIO()
                                    frame_pil.save(buf, format="JPEG")
                                    st.download_button(
                                        label=f"💾 Download Frame #{inc['frame_idx']}",
                                        data=buf.getvalue(),
                                        file_name=f"pothole_frame_{inc['frame_idx']}.jpg",
                                        mime="image/jpeg",
                                        key=f"dl_frame_{inc['frame_idx']}_{i}"
                                    )

                        # --------------------------------------------------
                        # INCIDENT LOG TABLE & CSV DOWNLOAD
                        # --------------------------------------------------
                        st.divider()
                        st.subheader("📋 Incident Frames Log Table")

                        log_rows = []
                        for inc in incidents:
                            log_rows.append({
                                "Frame #": inc["frame_idx"],
                                "Timestamp": inc["timestamp_str"],
                                "Damage Classes": ", ".join(inc["classes"]),
                                "Severity": inc["highest_severity"],
                                "Detections": inc["detections_count"],
                                "Max Confidence": f"{inc['max_confidence']:.2f}"
                            })

                        df_log = pd.DataFrame(log_rows)
                        st.dataframe(df_log, use_container_width=True, hide_index=True)

                        csv_data = df_log.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Damage Incident Log (CSV)",
                            data=csv_data,
                            file_name="pothole_incident_report.csv",
                            mime="text/csv",
                            type="secondary"
                        )

                    else:
                        st.info(
                            "No potholes or damage detected in the scanned video frames under the current confidence threshold."
                        )

                    # --------------------------------------------------
                    # DOWNLOAD FULL ANNOTATED VIDEO
                    # --------------------------------------------------
                    if res["output_path"] and Path(res["output_path"]).exists():
                        st.divider()
                        st.subheader("🎥 Download Annotated Video")
                        with open(res["output_path"], "rb") as vf:
                            st.download_button(
                                label="⬇️ Download Full Annotated Video (.mp4)",
                                data=vf.read(),
                                file_name="annotated_road_damage_video.mp4",
                                mime="video/mp4",
                                type="primary"
                            )

                except Exception as vid_err:
                    st.error(f"Video processing error: {vid_err}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Urban Road Damage Detection | "
    "YOLO11n | RDD2020 + RDD2022"
)
