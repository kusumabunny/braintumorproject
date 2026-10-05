import io
import zipfile

import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_URL = "http://127.0.0.1:8000"
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
# SIMPLE PROFESSIONAL THEME
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

st.subheader(
    "AI-Assisted Brain MRI Classification"
)

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
# PREDICTION FUNCTION
# ============================================================


def predict_file(
    file_bytes,
    filename,
    content_type,
):
    """Send an image to the FastAPI backend."""

    response = requests.post(
        PREDICT_URL,
        files={
            "file": (
                filename,
                io.BytesIO(file_bytes),
                content_type,
            )
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


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
                len(uploaded_file.getvalue())
                / 1024
            )

            st.write(
                f"**Filename:** {uploaded_file.name}"
            )

            st.write(
                f"**File size:** {file_size:.1f} KB"
            )

            st.write(
                f"**Format:** {uploaded_file.type}"
            )

        st.write("")

        if st.button(
            "🔍 Analyze MRI",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing MRI image..."
            ):

                try:

                    result = predict_file(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        uploaded_file.type,
                    )

                    st.session_state[
                        "single_result"
                    ] = result

                except requests.exceptions.ConnectionError:

                    st.error(
                        "❌ Cannot connect to FastAPI. "
                        "Make sure the backend is running."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "❌ Prediction service timed out."
                    )

                except requests.exceptions.HTTPError as error:

                    st.error(
                        f"❌ Backend error: {error}"
                    )

                except Exception as error:

                    st.error(
                        f"❌ Unexpected error: {error}"
                    )


    # ========================================================
    # RESULT
    # ========================================================

    if "single_result" in st.session_state:

        result = st.session_state["single_result"]

        final_prediction = result[
            "final_prediction"
        ]

        final_confidence = float(
            result["final_confidence"]
        )

        st.divider()

        st.header("🎯 Prediction Result")

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if (
            final_confidence
            < confidence_threshold
        ):

            st.warning(
                f"⚠️ **Needs Review** — "
                f"confidence is {final_confidence:.2f}%, "
                f"below the {confidence_threshold}% threshold."
            )

        else:

            st.success(
                f"✅ **High Confidence** — "
                f"{final_confidence:.2f}% confidence"
            )

        # ----------------------------------------------------
        # FINAL PREDICTION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CONFIDENCE BAR
        # ----------------------------------------------------

        st.write("**Confidence Level**")

        st.progress(
            min(
                final_confidence / 100,
                1.0,
            )
        )

        # ----------------------------------------------------
        # MODEL DETAILS
        # ----------------------------------------------------

        st.subheader("🔬 Model Assessment")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Binary Classification",
                result[
                    "binary_prediction"
                ],
                f'{result["binary_confidence"]:.2f}% confidence',
            )

        with col2:

            st.metric(
                "Tumor Classification",
                result[
                    "tumor_type"
                ],
                f'{result["tumor_confidence"]:.2f}% confidence',
            )

        with col3:

            if (
                final_confidence
                < confidence_threshold
            ):
                status = "Needs Review"
            else:
                status = "High Confidence"

            st.metric(
                "Review Status",
                status,
                f"Threshold: {confidence_threshold}%",
            )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.subheader("📋 Analysis Summary")

        if final_prediction == "No Tumor":

            st.info(
                f"The model classified the uploaded MRI "
                f"as **No Tumor** with "
                f"{final_confidence:.2f}% confidence."
            )

        else:

            st.info(
                f"The model classified the uploaded MRI "
                f"as **{final_prediction}** with "
                f"{final_confidence:.2f}% confidence."
            )


# ============================================================
# BULK ZIP TAB
# ============================================================

with bulk_tab:

    st.header("Bulk ZIP Analysis")

    st.info(
        "Upload a ZIP archive containing multiple MRI "
        "images. Each image will be classified individually."
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
                    io.BytesIO(
                        zip_file.getvalue()
                    )
                ) as archive:

                    image_files = [
                        name
                        for name in archive.namelist()
                        if name.lower().endswith(
                            (
                                ".jpg",
                                ".jpeg",
                                ".png",
                            )
                        )
                        and not name.endswith("/")
                    ]

                    if not image_files:

                        st.warning(
                            "No JPG, JPEG, or PNG "
                            "images were found."
                        )

                    else:

                        st.info(
                            f"Found {len(image_files)} "
                            f"image(s). Processing..."
                        )

                        progress = st.progress(0)

                        for index, filename in enumerate(
                            image_files
                        ):

                            try:

                                image_bytes = archive.read(
                                    filename
                                )

                                result = predict_file(
                                    image_bytes,
                                    filename,
                                    "image/jpeg",
                                )

                                confidence = float(
                                    result[
                                        "final_confidence"
                                    ]
                                )

                                review_status = (
                                    "Needs Review"
                                    if confidence
                                    < confidence_threshold
                                    else "OK"
                                )

                                results.append(
                                    {
                                        "Filename": filename,
                                        "Prediction": result[
                                            "final_prediction"
                                        ],
                                        "Confidence (%)": confidence,
                                        "Binary Result": result[
                                            "binary_prediction"
                                        ],
                                        "Tumor Type": result[
                                            "tumor_type"
                                        ],
                                        "Review Status": review_status,
                                    }
                                )

                            except Exception as error:

                                results.append(
                                    {
                                        "Filename": filename,
                                        "Prediction": "Error",
                                        "Confidence (%)": 0.0,
                                        "Binary Result": "-",
                                        "Tumor Type": "-",
                                        "Review Status": str(
                                            error
                                        ),
                                    }
                                )

                            progress.progress(
                                (index + 1)
                                / len(image_files)
                            )

                if results:

                    st.session_state[
                        "bulk_results"
                    ] = pd.DataFrame(results)

                    st.success(
                        f"✅ Processed "
                        f"{len(results)} image(s) successfully."
                    )

            except zipfile.BadZipFile:

                st.error(
                    "❌ The uploaded file is not "
                    "a valid ZIP archive."
                )

            except Exception as error:

                st.error(
                    f"❌ ZIP processing failed: {error}"
                )


    # ========================================================
    # BULK RESULTS
    # ========================================================

    if "bulk_results" in st.session_state:

        df = st.session_state[
            "bulk_results"
        ]

        st.divider()

        st.header("📊 Analysis Results")

        total = len(df)

        successful = int(
            (
                df["Prediction"]
                != "Error"
            ).sum()
        )

        needs_review = int(
            (
                df["Review Status"]
                == "Needs Review"
            ).sum()
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