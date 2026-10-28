"""
app.py — Streamlit Web UI for AI Property Video Generator.

Provides an intuitive web interface for agents to upload property details,
images, choose narration/music/LLM settings, preview generated videos,
and download final MP4s and scripts.
"""

import os
import re
import json
import shutil
import tempfile
from pathlib import Path

import streamlit as st

# Load environment (.env) if present
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import run_pipeline, PipelineOptions
from src.logger import setup_logger

logger = setup_logger()

BASE_DIR = Path(__file__).parent.resolve()
SAMPLE_HOUSE_1 = BASE_DIR / "input" / "samples" / "house_1"


def sanitize_filename(name: str) -> str:
    """Sanitize filename to keep alphanumeric, dashes, dots, and underscores."""
    clean = re.sub(r"[^\w\-.]", "_", name)
    return clean.strip("._") or "image"


def load_sample_house_1():
    """Prefill session state with house_1 sample data."""
    st.session_state["prop_name"] = "Maple Residency"
    st.session_state["prop_location"] = "Sector 150, Noida"
    st.session_state["prop_price"] = "8500000"
    st.session_state["prop_type_select"] = "Independent House"
    st.session_state["prop_type_other"] = ""
    st.session_state["prop_features"] = "Spacious kitchen\nComfortable bedroom\nClean bathroom"
    st.session_state["prop_cta"] = "Book a site visit today"
    st.session_state["prop_contact"] = "+91-00000-00000"
    st.session_state["prop_tagline"] = "Where every room feels like home"
    st.session_state["use_sample_images"] = True


def main():
    st.set_page_config(
        page_title="AI Property Video Generator",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎬 AI Property Video Generator")
    st.markdown(
        "Generate professional, branded real estate promo videos with AI-powered scripts, "
        "realistic narration, and background music in seconds."
    )

    # ── Sidebar / Quick Actions ─────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("Quickly test with bundled sample data:")
        if st.button("📋 Load Sample (Maple Residency)", use_container_width=True):
            load_sample_house_1()
            st.success("Sample data loaded!")

        st.divider()
        st.subheader("Capabilities")
        has_api_key = bool(os.getenv("LLM_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip())
        if has_api_key:
            st.info("✓ LLM API Key detected (AI Script generation ready)")
        else:
            st.warning("⚠ No LLM API Key detected (Running in deterministic template mode)")

    # ── Initialize session state defaults ───────────────────────────────────
    defaults = {
        "prop_name": "",
        "prop_location": "",
        "prop_price": "",
        "prop_type_select": "Apartment",
        "prop_type_other": "",
        "prop_features": "",
        "prop_cta": "Book a site visit today",
        "prop_contact": "",
        "prop_tagline": "",
        "use_sample_images": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Form Inputs ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("🏠 Property Details")
        name = st.text_input("Property Name *", value=st.session_state["prop_name"])
        location = st.text_input("Location / City *", value=st.session_state["prop_location"])
        price = st.text_input("Price * (e.g. 8500000 or ₹85 Lakhs)", value=st.session_state["prop_price"])

        type_options = ["Apartment", "Villa", "Independent House", "Penthouse", "Studio", "Commercial", "Other"]
        current_type_idx = type_options.index(st.session_state["prop_type_select"]) if st.session_state["prop_type_select"] in type_options else 0
        prop_type_sel = st.selectbox("Property Type *", type_options, index=current_type_idx)

        if prop_type_sel == "Other":
            prop_type = st.text_input("Custom Property Type *", value=st.session_state["prop_type_other"])
        else:
            prop_type = prop_type_sel

        features_raw = st.text_area(
            "Features * (One feature per line)",
            value=st.session_state["prop_features"],
            height=120,
            help="Enter 1 to 6 key highlights of the home (e.g. Spacious kitchen)",
        )

        cta_text = st.text_input("Call to Action *", value=st.session_state["prop_cta"])
        contact = st.text_input("Contact Info (Optional)", value=st.session_state["prop_contact"])
        tagline = st.text_input("Tagline (Optional)", value=st.session_state["prop_tagline"])

    with col_right:
        st.subheader("📸 Property Photos")
        st.markdown("Upload between **4 and 8 photos** (JPG, PNG, or WebP).")

        uploaded_files = st.file_uploader(
            "Choose photos",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Recommended: exterior/frontal, kitchen, bedroom, bathroom photos.",
        )

        # Handle sample images fallback
        using_sample_photos = False
        sample_images_list: list[Path] = []
        if (not uploaded_files) and st.session_state.get("use_sample_images"):
            sample_dir = SAMPLE_HOUSE_1 / "images"
            if sample_dir.exists():
                sample_images_list = sorted(
                    [p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
                )
                if sample_images_list:
                    using_sample_photos = True
                    st.info(f"Using {len(sample_images_list)} bundled sample images from House 1.")

        # Show thumbnail previews and counts
        total_photos = len(uploaded_files) if uploaded_files else len(sample_images_list)
        if total_photos == 0:
            st.info("👉 Please upload 4 to 8 photos, or click 'Load Sample'.")
        elif total_photos < 4:
            st.warning(f"⚠ {total_photos} photo(s) selected. At least 4 photos are recommended for optimal video pacing.")
        elif total_photos > 8:
            st.warning(f"⚠ {total_photos} photos selected. Only the first 8 photos will be included in the video.")
        else:
            st.success(f"✓ {total_photos} photos ready.")

        # Display thumbnails
        if uploaded_files:
            thumb_cols = st.columns(min(4, len(uploaded_files)))
            for idx, file in enumerate(uploaded_files[:8]):
                with thumb_cols[idx % len(thumb_cols)]:
                    st.image(file, caption=f"Photo {idx+1}", use_container_width=True)
        elif sample_images_list:
            thumb_cols = st.columns(min(4, len(sample_images_list)))
            for idx, img_path in enumerate(sample_images_list[:8]):
                with thumb_cols[idx % len(thumb_cols)]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)

        st.divider()
        st.subheader("🎛️ Video & Audio Settings")
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            use_ai_script = st.checkbox(
                "Use AI Script Generation",
                value=has_api_key,
                disabled=not has_api_key,
                help="Requires GEMINI_API_KEY or LLM_API_KEY. Disabled in template mode." if not has_api_key else "Use Gemini to generate natural real estate copy.",
            )
            enable_narration = st.checkbox("Include Voiceover Narration", value=True)
        with opt_col2:
            enable_music = st.checkbox("Include Background Music", value=True)
            resolution = st.selectbox("Resolution", ["720p (1280x720)", "1080p (1920x1080)"], index=0)

    # ── Generation Section ──────────────────────────────────────────────────
    st.divider()
    generate_btn = st.button("🚀 Generate Property Video", type="primary", use_container_width=True)

    if generate_btn:
        # Pre-validation
        features_list = [f.strip() for f in features_raw.splitlines() if f.strip()]
        val_errors: list[str] = []

        if not name.strip():
            val_errors.append("Property Name is required.")
        if not location.strip():
            val_errors.append("Location is required.")
        if not price.strip():
            val_errors.append("Price is required.")
        if not prop_type.strip():
            val_errors.append("Property Type is required.")
        if not features_list:
            val_errors.append("At least one feature is required.")
        if not cta_text.strip():
            val_errors.append("Call to Action is required.")

        if not uploaded_files and not sample_images_list:
            val_errors.append("Please upload at least 4 photos (or load the sample).")

        if val_errors:
            for err in val_errors:
                st.error(f"❌ {err}")
            return

        # Prepare property input dict
        prop_dict = {
            "name": name.strip(),
            "location": location.strip(),
            "price": price.strip(),
            "property_type": prop_type.strip(),
            "features": features_list,
            "cta_text": cta_text.strip(),
        }
        if contact.strip():
            prop_dict["contact"] = contact.strip()
        if tagline.strip():
            prop_dict["tagline"] = tagline.strip()

        # Build pipeline options
        chosen_size = "1080p" if "1080p" in resolution else "720p"
        options = PipelineOptions(
            use_llm=use_ai_script if has_api_key else False,
            size=chosen_size,
            no_audio=not (enable_narration or enable_music),
            no_music=not enable_music,
            no_narration=not enable_narration,
        )

        # Prepare progress tracking UI
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def on_progress(stage: str, frac: float, message: str):
            progress_bar.progress(min(1.0, max(0.0, frac)))
            stage_icon = {
                "validation": "🔍",
                "script": "✍️",
                "audio": "🎙️",
                "render": "🎞️",
                "complete": "🎉",
            }.get(stage, "⏳")
            status_box.markdown(f"**{stage_icon} [{stage.title()}]** {message}")

        # Run pipeline in a clean temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp(prefix="realty_app_")
        try:
            images_to_process: list[Path] = []
            if uploaded_files:
                for idx, file in enumerate(uploaded_files):
                    clean_name = sanitize_filename(file.name)
                    # Use index prefix to preserve upload order when room keywords are absent
                    dest_file = Path(temp_dir) / f"{idx:02d}_{clean_name}"
                    with open(dest_file, "wb") as f_out:
                        f_out.write(file.getbuffer())
                    images_to_process.append(dest_file)
            else:
                images_to_process = sample_images_list

            # Execute unified pipeline
            with st.spinner("Processing video..."):
                result = run_pipeline(
                    property_input=prop_dict,
                    images_input=images_to_process,
                    options=options,
                    progress_cb=on_progress,
                )

            # Display any warnings
            if result.warnings:
                with st.expander(f"⚠️ Validation & Pipeline Notices ({len(result.warnings)})", expanded=False):
                    for w in result.warnings:
                        st.warning(w)

            # Display errors if any
            if not result.ok or result.errors:
                for err in result.errors:
                    st.error(f"❌ {err}")
                return

            # Display success and results
            st.success("🎉 Video successfully generated!")

            # Metric summary
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Duration", f"{result.plan.total_duration}s")
            m2.metric("Resolution", chosen_size)
            m3.metric("File Size", f"{result.file_size / (1024 * 1024):.1f} MB")
            m4.metric("Render Time", f"{result.timings.get('render', 0.0):.1f}s")

            # Scene Plan Table
            st.subheader("📋 Scene Storyboard & Narration Plan")
            table_data = []
            for idx, scene in enumerate(result.plan.scenes, start=1):
                table_data.append({
                    "Scene": idx,
                    "Type": scene.scene_type.title(),
                    "Caption (On Screen)": scene.caption.replace("\n", " | "),
                    "Voiceover Narration": scene.voiceover,
                    "Duration": f"{scene.duration:.1f}s",
                })
            st.table(table_data)

            # Video Preview
            st.subheader("🎥 Video Preview")
            if result.video_path and result.video_path.exists():
                with open(result.video_path, "rb") as vf:
                    video_bytes = vf.read()
                st.video(video_bytes)

                # Download buttons
                down_col1, down_col2 = st.columns(2)
                with down_col1:
                    st.download_button(
                        label="⬇️ Download Video (MP4)",
                        data=video_bytes,
                        file_name=result.video_path.name,
                        mime="video/mp4",
                        use_container_width=True,
                    )
                with down_col2:
                    script_json = json.dumps(result.plan.to_dict(), indent=2, ensure_ascii=False)
                    st.download_button(
                        label="⬇️ Download Script (JSON)",
                        data=script_json,
                        file_name="script.json",
                        mime="application/json",
                        use_container_width=True,
                    )

        except Exception as exc:
            logger.exception("Streamlit app error during generation: %s", exc)
            st.error(f"❌ An error occurred during generation: {exc}")
            st.info("Check logs/app.log for additional debugging details.")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
