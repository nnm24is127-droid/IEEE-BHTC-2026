"""
ParkMod – app.py
Streamlit Dashboard with interactive Zone Designer.

Tabs:
  1. Dashboard       – live metrics
  2. Zone Designer   – create / edit / delete / preview ROI zones
  3. Violations      – table of all violations
  4. Evidence        – gallery of saved evidence images
  5. Live Detector   – run inference in the browser
  6. Settings        – quick config tweaks
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas

    _HAS_DRAWABLE_CANVAS = True
except ImportError:
    _HAS_DRAWABLE_CANVAS = False

# ── Streamlit page config (must be first st call) ────────────────
st.set_page_config(
    page_title="ParkMod Dashboard",
    layout="wide",
    page_icon="🚫",
)

from src.config import (
    CSV_REPORT_PATH,
    DEFAULT_VIOLATION_THRESHOLD_SEC,
    EVIDENCE_DIR,
    INPUT_VIDEO_DIR,
    OUTPUT_FRAME_DIR,
    PROJECT_NAME,
    PROJECT_VERSION,
    WEBCAM_INDEX,
    ZONES_FILE,
)
from src.main import ParkModPipeline
from src.roi_manager import ROIManager, Zone


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _load_violations_df() -> pd.DataFrame:
    """Load the violations CSV into a DataFrame."""
    if Path(CSV_REPORT_PATH).exists():
        try:
            return pd.read_csv(CSV_REPORT_PATH, comment="#")
        except Exception:
            pass
    return pd.DataFrame()


def _parse_point(text: str) -> list:
    """Parse 'x,y' → [x, y] as ints."""
    parts = text.replace(" ", "").split(",")
    return [int(parts[0]), int(parts[1])]


def _first_frame_from_video(path: str) -> np.ndarray | None:
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _apply_live_threshold_override(pipe: ParkModPipeline, threshold_sec: float) -> int:
    """Apply threshold override to currently active zones for this run only."""
    changed = 0
    for zone in pipe.roi.active_zones:
        zone.threshold = float(threshold_sec)
        changed += 1
    return changed


def _extract_polygon_from_canvas(
    canvas_json: dict,
    sx: float,
    sy: float,
) -> list | None:
    """Convert latest canvas shape to polygon in original image coordinates."""
    if not canvas_json:
        return None

    objects = canvas_json.get("objects", [])
    if not objects:
        return None

    obj = objects[-1]
    otype = obj.get("type", "")

    if otype == "rect":
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
        height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))
        if width <= 1 or height <= 1:
            return None

        x1 = int(left * sx)
        y1 = int(top * sy)
        x2 = int((left + width) * sx)
        y2 = int((top + height) * sy)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    if otype in ("polygon", "path"):
        points = []
        if otype == "polygon":
            for p in obj.get("points", []):
                px = float(obj.get("left", 0)) + float(p.get("x", 0)) * float(
                    obj.get("scaleX", 1.0)
                )
                py = float(obj.get("top", 0)) + float(p.get("y", 0)) * float(
                    obj.get("scaleY", 1.0)
                )
                points.append([int(px * sx), int(py * sy)])
        else:
            for cmd in obj.get("path", []):
                if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M", "L"):
                    points.append([int(float(cmd[1]) * sx), int(float(cmd[2]) * sy)])

        dedup = []
        for pt in points:
            if not dedup or dedup[-1] != pt:
                dedup.append(pt)
        if len(dedup) >= 3:
            return dedup

    return None


# ═══════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <h1 style='text-align:center;'>🚀 {PROJECT_NAME}</h1>
    <p style='text-align:center;color:#888;'>
        v{PROJECT_VERSION} — YOLOv8 ByteTrack · Dynamic ROI Zones · 2-Stage ALPR
    </p>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📊 Dashboard",
    "🎨 Zone Designer",
    "📋 Violations",
    "📸 Evidence",
    "🎥 Live Detector",
    "⚙️ Settings",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 – Dashboard
# ═══════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("System Overview")
    df = _load_violations_df()
    mgr = ROIManager()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Violations", len(df) if not df.empty else 0)
    c2.metric("Total Zones", len(mgr.zones))
    c3.metric("Active Zones", len(mgr.active_zones))
    c4.metric(
        "Unique Plates",
        df["plate_number"].nunique() if not df.empty else 0,
    )

    if not df.empty:
        st.subheader("Recent Violations")
        st.dataframe(df.tail(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 – Zone Designer
# ═══════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("🎨 Interactive ROI Zone Designer")
    mgr = ROIManager()  # fresh load

    if "drawn_polygon" not in st.session_state:
        st.session_state.drawn_polygon = None

    # ── Reference frame for preview ───────────────────────────────
    st.subheader("① Upload / Select Reference Frame")
    st.caption(
        "Upload an image from your video or camera so you can "
        "preview where the zone falls on the actual scene."
    )
    ref_file = st.file_uploader(
        "Upload reference image (screenshot / first frame)",
        type=["jpg", "jpeg", "png", "bmp"],
        key="zone_ref_upload",
    )
    ref_frame = None
    if ref_file is not None:
        file_bytes = np.frombuffer(ref_file.read(), np.uint8)
        ref_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(
            cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB),
            caption=f"Reference frame — {ref_frame.shape[1]}×{ref_frame.shape[0]}",
            use_container_width=True,
        )

        if _HAS_DRAWABLE_CANVAS:
            st.subheader("② Draw Zone Directly (Drag-and-Draw)")
            st.caption(
                "Draw a rectangle or polygon on the reference image. "
                "The latest shape can be used directly while saving the zone."
            )

            draw_cols = st.columns(2)
            with draw_cols[0]:
                canvas_mode = st.selectbox(
                    "Draw mode",
                    ["rect", "polygon"],
                    key="zone_canvas_mode",
                )
            with draw_cols[1]:
                st.write("")
                if st.button("🧹 Clear drawn shape", key="clear_zone_shape"):
                    st.session_state.drawn_polygon = None

            h, w = ref_frame.shape[:2]
            max_canvas_w = 1000
            canvas_w = min(max_canvas_w, w)
            canvas_h = int((h / w) * canvas_w)

            display_img = cv2.resize(
                ref_frame,
                (canvas_w, canvas_h),
                interpolation=cv2.INTER_AREA,
            )

            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 100, 0.25)",
                stroke_width=3,
                stroke_color="#00FF64",
                background_image=Image.fromarray(
                    cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                ),
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode=canvas_mode,
                key="zone_draw_canvas",
            )

            drawn_poly = _extract_polygon_from_canvas(
                canvas_result.json_data,
                sx=(w / canvas_w),
                sy=(h / canvas_h),
            )
            if drawn_poly:
                st.session_state.drawn_polygon = drawn_poly
                st.success(
                    f"Captured shape with {len(drawn_poly)} points. "
                    "You can use it while saving the zone below."
                )
                st.code(
                    "\n".join(f"{p[0]},{p[1]}" for p in drawn_poly),
                    language="text",
                )
        else:
            st.info(
                "Install `streamlit-drawable-canvas` to enable drag-and-draw zones."
            )

    st.markdown("---")

    # ── Create new zone ───────────────────────────────────────────
    st.subheader("③ Create New Zone")

    with st.form("create_zone_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)
        with col_a:
            z_name = st.text_input("Zone Name", placeholder="e.g. VIP Drop-off")
            z_type = st.selectbox("Zone Type", ["rectangle", "polygon"])
            z_thresh = st.number_input(
                "Violation Threshold (seconds)", value=10.0,
                min_value=1.0, step=1.0,
            )
        with col_b:
            z_location = st.text_input("Location", placeholder="e.g. Main Gate")
            z_department = st.text_input("Department", value="Campus Security")
            z_color_hex = st.color_picker("Zone Overlay Color", "#00FF64")

        use_drawn = st.checkbox(
            "Use last drawn shape from canvas",
            value=bool(st.session_state.drawn_polygon),
            help="If checked and a shape is available, manual coordinates are ignored.",
        )

        st.markdown("**Define Zone Coordinates**")
        if z_type == "rectangle":
            rc1, rc2 = st.columns(2)
            with rc1:
                rect_x1 = st.number_input("Top-Left X", value=100, step=10)
                rect_y1 = st.number_input("Top-Left Y", value=300, step=10)
            with rc2:
                rect_x2 = st.number_input("Bottom-Right X", value=1100, step=10)
                rect_y2 = st.number_input("Bottom-Right Y", value=650, step=10)
        else:
            st.info(
                "Enter polygon vertices as **x,y** pairs "
                "(one per line).  Minimum 3 points."
            )
            poly_raw = st.text_area(
                "Polygon points (one x,y per line)",
                value="100,300\n1100,300\n1100,650\n100,650",
                height=120,
            )

        submitted = st.form_submit_button("💾 Save Zone")

    if submitted:
        # Build polygon list
        if use_drawn and st.session_state.drawn_polygon:
            polygon = st.session_state.drawn_polygon
            if len(polygon) == 4:
                z_type = "rectangle"
            else:
                z_type = "polygon"
        elif z_type == "rectangle":
            polygon = [
                [rect_x1, rect_y1],
                [rect_x2, rect_y1],
                [rect_x2, rect_y2],
                [rect_x1, rect_y2],
            ]
        else:
            try:
                lines = [
                    ln.strip() for ln in poly_raw.strip().splitlines()
                    if ln.strip()
                ]
                polygon = [_parse_point(ln) for ln in lines]
                if len(polygon) < 3:
                    st.error("Polygon needs at least 3 points.")
                    polygon = None
            except Exception as exc:
                st.error(f"Could not parse polygon points: {exc}")
                polygon = None

        if polygon and z_name:
            # Convert hex colour to BGR
            hex_c = z_color_hex.lstrip("#")
            r, g, b = int(hex_c[:2], 16), int(hex_c[2:4], 16), int(hex_c[4:], 16)
            bgr = [b, g, r]

            new_zone = Zone(
                id=f"Z-{uuid.uuid4().hex[:6].upper()}",
                name=z_name,
                zone_type=z_type,
                polygon=polygon,
                threshold=z_thresh,
                location=z_location,
                department=z_department,
                active=True,
                color=bgr,
            )
            mgr.add_zone(new_zone)
            st.success(f"✅ Zone **{z_name}** saved!  (ID: {new_zone.id})")

            # Show preview if reference frame available
            if ref_frame is not None:
                preview = mgr.draw_zone_on_image(ref_frame.copy(), new_zone)
                st.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    caption="Zone preview on reference frame",
                    use_container_width=True,
                )
            st.rerun()

    st.markdown("---")

    # ── List / manage existing zones ──────────────────────────────
    st.subheader("④ Manage Saved Zones")

    if not mgr.zones:
        st.info("No zones saved yet.  Create one above ☝️")
    else:
        for zone in mgr.zones:
            status_label = "🟢 Active" if zone.active else "🔴 Inactive"
            with st.expander(
                f"{status_label}  **{zone.name}**  ·  {zone.zone_type}"
                f"  ·  {zone.threshold}s  ·  {zone.location}"
                f"  ·  `{zone.id}`",
                expanded=False,
            ):
                # Show zone details
                st.json(zone.to_dict())

                # Preview button
                if ref_frame is not None:
                    if st.button(f"👁️ Preview", key=f"prev_{zone.id}"):
                        preview = mgr.draw_zone_on_image(ref_frame.copy(), zone)
                        st.image(
                            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                            caption=f"Preview: {zone.name}",
                            use_container_width=True,
                        )

                # Action buttons
                btn_cols = st.columns(3)

                # Toggle active
                with btn_cols[0]:
                    label = "⏸️ Deactivate" if zone.active else "▶️ Activate"
                    if st.button(label, key=f"toggle_{zone.id}"):
                        mgr.toggle_zone(zone.id)
                        st.rerun()

                # Edit threshold
                with btn_cols[1]:
                    new_thresh = st.number_input(
                        "New Threshold (s)",
                        value=zone.threshold,
                        min_value=1.0,
                        step=1.0,
                        key=f"thresh_{zone.id}",
                    )
                    if st.button("✏️ Update Threshold", key=f"upd_{zone.id}"):
                        mgr.update_zone(zone.id, {"threshold": new_thresh})
                        st.success(f"Threshold updated to {new_thresh}s")
                        st.rerun()

                # Delete
                with btn_cols[2]:
                    if st.button("🗑️ Delete Zone", key=f"del_{zone.id}"):
                        mgr.delete_zone(zone.id)
                        st.warning(f"Zone **{zone.name}** deleted.")
                        st.rerun()

    # ── Full preview of all active zones ──────────────────────────
    if ref_frame is not None and mgr.active_zones:
        st.markdown("---")
        st.subheader("⑤ All Active Zones Preview")
        combined = mgr.draw_zones(ref_frame.copy())
        st.image(
            cv2.cvtColor(combined, cv2.COLOR_BGR2RGB),
            caption="All active zones overlaid on reference frame",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 3 – Violations
# ═══════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("📋 Violation Records")
    df = _load_violations_df()
    if df.empty:
        st.info("No violations recorded yet.")
    else:
        st.dataframe(df, use_container_width=True, height=500)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_bytes,
            file_name="violations_report.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════
# TAB 4 – Evidence Gallery
# ═══════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("📸 Evidence Gallery")
    evidence_dir = Path(EVIDENCE_DIR)
    if evidence_dir.exists():
        images = sorted(evidence_dir.glob("*.jpg"), reverse=True)
        if not images:
            st.info("No evidence images captured yet.")
        else:
            cols = st.columns(4)
            for i, img_path in enumerate(images):
                with cols[i % 4]:
                    st.image(
                        Image.open(img_path),
                        caption=img_path.name,
                        use_container_width=True,
                    )
    else:
        st.info("Evidence directory does not exist yet.")


# ═══════════════════════════════════════════════════════════════
# TAB 5 – Live Detector
# ═══════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("🎥 Live Detection Stream")

    st.subheader("Upload Video (Drag & Drop)")
    uploaded_video = st.file_uploader(
        "Drop video file here (mp4, avi, mov, mkv, mpeg, mpg)",
        type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"],
        key="live_video_upload",
    )

    selected_source = "0"
    if uploaded_video is not None:
        save_path = Path(INPUT_VIDEO_DIR) / uploaded_video.name
        with open(save_path, "wb") as fh:
            fh.write(uploaded_video.getbuffer())

        st.success(f"Uploaded: {uploaded_video.name}")
        st.video(uploaded_video)
        selected_source = str(save_path)

    src_input = st.text_input(
        "Video path or webcam index",
        value=selected_source,
        key="live_source",
    )

    st.subheader("Detection Timing")
    use_threshold_override = st.checkbox(
        "Override violation threshold for this run",
        value=False,
        help="Apply one threshold value to all active zones while running inference.",
    )
    threshold_override = st.number_input(
        "Violation threshold (seconds)",
        min_value=1.0,
        max_value=300.0,
        value=float(DEFAULT_VIOLATION_THRESHOLD_SEC),
        step=1.0,
        disabled=not use_threshold_override,
    )

    frame_stride = st.number_input(
        "Process every Nth frame (1 = all frames)",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Higher values run faster but may reduce detection accuracy.",
    )

    col_start, col_stop = st.columns(2)
    start_btn = col_start.button("▶️ Start Inference")
    stop_btn = col_stop.button("⏹️ Stop")

    if start_btn:
        stframe = st.empty()
        status_box = st.empty()
        progress = st.progress(0)
        src = int(src_input) if src_input.isdigit() else src_input
        pipe = ParkModPipeline(demo_mode=False)

        if use_threshold_override:
            changed = _apply_live_threshold_override(pipe, threshold_override)
            status_box.info(
                f"Applied threshold override: {threshold_override:.1f}s "
                f"to {changed} active zone(s)."
            )

        if not pipe.alpr.is_plate_detector_ready():
            st.warning(
                "Plate detector model is not loaded (models/plate_detector.pt missing). "
                "System is using heuristic plate localisation, which is less accurate."
            )
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            st.error("Could not open the selected video source.")
            st.stop()

        use_video_time = not str(src_input).isdigit()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = 0

        writer = None
        output_video_path = None
        if use_video_time:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w > 0 and h > 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_video_path = Path(OUTPUT_FRAME_DIR) / f"violation_output_{ts}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        while cap.isOpened():
            if stop_btn:
                status_box.warning("Inference stopped by user.")
                break

            ret, frame = cap.read()
            if not ret:
                st.info("End of video stream.")
                break

            if frame_stride > 1 and (frame_idx % frame_stride) != 0:
                frame_idx += 1
                continue

            if use_video_time:
                pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if pos_sec <= 0:
                    pos_sec = frame_idx / fps
                current_time = pos_sec
            else:
                current_time = time.time()

            annotated = pipe.process_frame(frame, current_time)

            if writer is not None:
                writer.write(annotated)

            frame_idx += 1

            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0))

            status_box.caption(
                f"Frame {frame_idx}"
                + (f" / {total_frames}" if total_frames > 0 else "")
                + f"  |  Sim Time: {current_time:.2f}s"
            )

            stframe.image(
                annotated, channels="BGR", use_container_width=True
            )

        cap.release()
        if writer is not None:
            writer.release()

        if output_video_path and output_video_path.exists():
            st.success(f"Processed video saved: {output_video_path.name}")
            st.video(str(output_video_path))


# ═══════════════════════════════════════════════════════════════
# TAB 6 – Settings
# ═══════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("⚙️ Quick Settings")

    st.subheader("Zone Data")
    if ZONES_FILE.exists():
        with open(ZONES_FILE, "r", encoding="utf-8") as fh:
            st.download_button(
                "📥 Export zones.json",
                data=fh.read(),
                file_name="zones.json",
                mime="application/json",
            )

    uploaded_zones = st.file_uploader(
        "📤 Import zones.json", type=["json"], key="import_zones"
    )
    if uploaded_zones is not None:
        try:
            data = json.load(uploaded_zones)
            with open(ZONES_FILE, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
            st.success("Zones imported successfully!  Reload the page to see changes.")
        except Exception as exc:
            st.error(f"Import failed: {exc}")

    st.markdown("---")
    st.subheader("Paths & Info")
    st.text(f"Project        : {PROJECT_NAME}")
    st.text(f"Version        : {PROJECT_VERSION}")
    st.text(f"Zones File     : {ZONES_FILE}")
    st.text(f"CSV Report     : {CSV_REPORT_PATH}")
    st.text(f"Evidence Dir   : {EVIDENCE_DIR}")
