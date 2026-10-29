import io
import os
import zipfile

import pandas as pd
import requests
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

# Local default; the Azure deployment sets BACKEND_URL explicitly.
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
).rstrip("/")

PREDICT_URL = f"{BACKEND_URL}/predict"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #102a43;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        height: 3rem;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #d9e2ec;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Brain Tumor MRI Classifier")
st.subheader("AI-Assisted Brain MRI Classification")
st.caption(
    "Deep Learning • ResNet18 • Binary + Multi-Class Classification"
)
st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ System Settings")

    st.write(
        "Configure the confidence threshold used to "
        "flag predictions for review."
    )

    confidence_threshold = st.slider(
        "Review Threshold",
        min_value=50,
        max_value=99,
        value=80,
        step=1,
    )

    st.metric(
        "Current Threshold",
        f"{confidence_threshold}%",
    )

    st.divider()
    st.subheader("Classification Models")

    st.write("🔬 **Binary Model**")
    st.caption("Tumor vs No Tumor")

    st.write("🧬 **Multi-Class Model**")
    st.caption(
        "Glioma • Meningioma • No Tumor • Pituitary"
    )

    st.divider()

    st.warning(
        "Educational / internship project. "
        "Not intended for clinical diagnosis."
    )


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_file(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> dict:
    """Send an image to the FastAPI backend."""
    response = requests.post(
        PREDICT_URL,
        files={
            "file": (
                filename,
                io.BytesIO(file_bytes),
                content_type or "application/octet-stream",
            )
        },
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise TypeError(
            "The backend returned an unexpected response format."
        )

    return result


def get_result_value(
    result: dict,
    key: str,
    default: str = "-",
) -> str:
    """Safely retrieve a display value from a prediction."""
    value = result.get(key, default)
    return str(value) if value is not None else default


def get_confidence(
    result: dict,
    key: str,
) -> float:
    """Retrieve a confidence value and validate its range."""
    confidence = float(result[key])

    if not 0 <= confidence <= 100:
        raise ValueError(
            f"{key} must be between 0 and 100."
        )

    return confidence


# ============================================================
# TABS
# ============================================================

single_tab, bulk_tab = st.tabs(
    [
        "🖼️ Single MRI Analysis",
        "📦 Bulk ZIP Analysis",
    ]
)


# ============================================================
# SINGLE MRI TAB
# ============================================================

with single_tab:
    st.header("Single MRI Analysis")

    st.info(
        "Upload one MRI image to perform binary tumor "
        "detection and multi-class tumor classification."
    )

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png"],
        key="single_upload",
    )

    if uploaded_file is not None:
        st.subheader("Uploaded MRI")

        image_col, details_col = st.columns(
            [1.4, 1],
            gap="large",
        )

        with image_col:
            st.image(
                uploaded_file,
                caption="MRI Image",
                use_container_width=True,
            )

        with details_col:
            st.subheader("File Information")

            file_size = (
                len(uploaded_file.getvalue()) / 1024
            )

            st.write(
                f"**Filename:** {uploaded_file.name}"
            )
            st.write(
                f"**File size:** {file_size:.1f} KB"
            )
            st.write(
                f"**Format:** {uploaded_file.type or 'Unknown'}"
            )

        st.write("")

        if st.button(
            "🔍 Analyze MRI",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Analyzing MRI image..."):
                try:
                    result = predict_file(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        uploaded_file.type or "application/octet-stream",
                    )

                    # Validate required result fields before saving.
                    if "final_prediction" not in result:
                        raise ValueError(
                            "Backend response is missing final_prediction."
                        )

                    get_confidence(
                        result,
                        "final_confidence",
                    )

                    st.session_state["single_result"] = result

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to FastAPI. "
                        "Check the backend URL and ensure "
                        "the backend is running."
                    )

                except requests.exceptions.Timeout:
                    st.error(
                        "❌ Prediction service timed out. "
                        "Please try again."
                    )

                except requests.exceptions.HTTPError as error:
                    st.error(
                        f"❌ Backend returned an HTTP error: {error}"
                    )

                except requests.exceptions.RequestException as error:
                    st.error(
                        f"❌ Prediction request failed: {error}"
                    )

                except (
                    ValueError,
                    KeyError,
                    TypeError,
                ) as error:
                    st.error(
                        f"❌ Invalid prediction response: {error}"
                    )

    # ========================================================
    # SINGLE RESULT
    # ========================================================

    if "single_result" in st.session_state:
        result = st.session_state["single_result"]

        try:
            final_prediction = get_result_value(
                result,
                "final_prediction",
            )

            final_confidence = get_confidence(
                result,
                "final_confidence",
            )

        except (KeyError, TypeError, ValueError) as error:
            st.error(
                f"Invalid prediction result: {error}"
            )

        else:
            st.divider()
            st.header("🎯 Prediction Result")

            if final_confidence < confidence_threshold:
                st.warning(
                    f"⚠️ **Needs Review** — confidence is "
                    f"{final_confidence:.2f}%, below the "
                    f"{confidence_threshold}% threshold."
                )
            else:
                st.success(
                    f"✅ **High Confidence** — "
                    f"{final_confidence:.2f}% confidence"
                )

            st.subheader("Final Classification")

            prediction_col, confidence_col = st.columns(2)

            with prediction_col:
                st.metric(
                    "Prediction",
                    final_prediction,
                )

            with confidence_col:
                st.metric(
                    "Confidence",
                    f"{final_confidence:.2f}%",
                )

            st.write("**Confidence Level**")

            st.progress(
                min(
                    max(final_confidence / 100, 0.0),
                    1.0,
                )
            )

            # ------------------------------------------------
            # MODEL DETAILS
            # ------------------------------------------------

            st.subheader("🔬 Model Assessment")

            col1, col2, col3 = st.columns(3)

            try:
                binary_confidence = get_confidence(
                    result,
                    "binary_confidence",
                )
            except (KeyError, TypeError, ValueError):
                binary_confidence = 0.0

            try:
                tumor_confidence = get_confidence(
                    result,
                    "tumor_confidence",
                )
            except (KeyError, TypeError, ValueError):
                tumor_confidence = 0.0

            with col1:
                st.metric(
                    "Binary Classification",
                    get_result_value(
                        result,
                        "binary_prediction",
                    ),
                    f"{binary_confidence:.2f}% confidence",
                )

            with col2:
                st.metric(
                    "Tumor Classification",
                    get_result_value(
                        result,
                        "tumor_type",
                    ),
                    f"{tumor_confidence:.2f}% confidence",
                )

            with col3:
                status = (
                    "Needs Review"
                    if final_confidence < confidence_threshold
                    else "High Confidence"
                )

                st.metric(
                    "Review Status",
                    status,
                    f"Threshold: {confidence_threshold}%",
                )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.subheader("📋 Analysis Summary")

            if final_prediction == "No Tumor":
                st.info(
                    "The model classified the uploaded image as "
                    f"**No Tumor** with {final_confidence:.2f}% "
                    "confidence. This is a model output, not "
                    "a medical diagnosis."
                )

            elif final_prediction == "Unrelated Image":
                st.warning(
                    "The model marked this upload as an unrelated "
                    "image. This result is not a medical assessment."
                )

            else:
                st.info(
                    f"The model classified the uploaded image as "
                    f"**{final_prediction}** with "
                    f"{final_confidence:.2f}% confidence. "
                    "This is a model output, not a medical diagnosis."
                )


# ============================================================
# BULK ZIP TAB
# ============================================================

with bulk_tab:
    st.header("Bulk ZIP Analysis")

    st.info(
        "Upload a ZIP archive containing multiple MRI images. "
        "Each image will be classified individually."
    )

    zip_file = st.file_uploader(
        "Choose a ZIP file",
        type=["zip"],
        key="bulk_upload",
    )

    if zip_file is not None:
        st.write(
            f"**Selected:** {zip_file.name}"
        )

        if st.button(
            "📊 Analyze ZIP",
            type="primary",
            use_container_width=True,
        ):
            results = []

            try:
                with zipfile.ZipFile(
                    io.BytesIO(zip_file.getvalue())
                ) as archive:

                    image_files = [
                        name
                        for name in archive.namelist()
                        if name.lower().endswith(
                            (".jpg", ".jpeg", ".png")
                        )
                        and not name.endswith("/")
                    ]

                    if not image_files:
                        st.warning(
                            "No JPG, JPEG, or PNG images were found."
                        )

                    else:
                        st.info(
                            f"Found {len(image_files)} image(s). "
                            "Processing..."
                        )

                        progress = st.progress(0)

                        for index, filename in enumerate(
                            image_files
                        ):
                            try:
                                image_bytes = archive.read(
                                    filename
                                )

                                extension = (
                                    filename.lower().rsplit(".", 1)[-1]
                                )

                                content_type = {
                                    "jpg": "image/jpeg",
                                    "jpeg": "image/jpeg",
                                    "png": "image/png",
                                }.get(
                                    extension,
                                    "application/octet-stream",
                                )

                                result = predict_file(
                                    image_bytes,
                                    filename,
                                    content_type,
                                )

                                confidence = get_confidence(
                                    result,
                                    "final_confidence",
                                )

                                review_status = (
                                    "Needs Review"
                                    if confidence < confidence_threshold
                                    else "OK"
                                )

                                results.append(
                                    {
                                        "Filename": filename,
                                        "Prediction": get_result_value(
                                            result,
                                            "final_prediction",
                                        ),
                                        "Confidence (%)": confidence,
                                        "Binary Result": get_result_value(
                                            result,
                                            "binary_prediction",
                                        ),
                                        "Tumor Type": get_result_value(
                                            result,
                                            "tumor_type",
                                        ),
                                        "Review Status": review_status,
                                    }
                                )

                            except requests.exceptions.RequestException as error:
                                results.append(
                                    {
                                        "Filename": filename,
                                        "Prediction": "Error",
                                        "Confidence (%)": 0.0,
                                        "Binary Result": "-",
                                        "Tumor Type": "-",
                                        "Review Status": (
                                            f"Request failed: {error}"
                                        ),
                                    }
                                )

                            except (
                                KeyError,
                                TypeError,
                                ValueError,
                                OSError,
                                RuntimeError,
                            ) as error:
                                results.append(
                                    {
                                        "Filename": filename,
                                        "Prediction": "Error",
                                        "Confidence (%)": 0.0,
                                        "Binary Result": "-",
                                        "Tumor Type": "-",
                                        "Review Status": str(error),
                                    }
                                )

                            progress.progress(
                                (index + 1) / len(image_files)
                            )

                if results:
                    st.session_state["bulk_results"] = pd.DataFrame(
                        results
                    )

                    successful_count = sum(
                        row["Prediction"] != "Error"
                        for row in results
                    )

                    st.success(
                        f"Processed {len(results)} image(s); "
                        f"{successful_count} prediction(s) completed."
                    )

            except zipfile.BadZipFile:
                st.error(
                    "❌ The uploaded file is not a valid ZIP archive."
                )

            except (OSError, RuntimeError, ValueError) as error:
                st.error(
                    f"❌ ZIP processing failed: {error}"
                )

    # ========================================================
    # BULK RESULTS
    # ========================================================

    if "bulk_results" in st.session_state:
        df = st.session_state["bulk_results"]

        st.divider()
        st.header("📊 Analysis Results")

        total = len(df)

        successful = int(
            (df["Prediction"] != "Error").sum()
        )

        needs_review = int(
            (df["Review Status"] == "Needs Review").sum()
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Images Processed",
                total,
            )

        with col2:
            st.metric(
                "Successful Predictions",
                successful,
            )

        with col3:
            st.metric(
                "Needs Review",
                needs_review,
            )

        st.write("")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv_data,
            file_name="brain_tumor_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 Brain Tumor MRI Classification | "
    "ResNet18 + FastAPI + Streamlit"
)

st.caption(
    "Educational / internship project. "
    "Not intended for clinical diagnosis."
)