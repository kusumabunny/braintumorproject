from io import BytesIO
import json
import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from azure.storage.blob import BlobServiceClient

from backend.predictor import predict_image


app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description="FastAPI backend for Brain Tumor MRI classification",
    version="1.0.0",
)


# ============================================================
# AZURE BLOB STORAGE
# ============================================================

AZURE_STORAGE_CONNECTION_STRING = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING"
)

BLOB_CONTAINER_NAME = "prediction-logs"


def save_prediction_log(filename, result):
    """Save prediction metadata as a JSON file in Azure Blob Storage."""

    if not AZURE_STORAGE_CONNECTION_STRING:
        return

    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )

        container_client = blob_service_client.get_container_client(
            BLOB_CONTAINER_NAME
        )

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            **result,
        }

        blob_name = (
            f"prediction_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            f"_{uuid4().hex[:8]}.json"
        )

        blob_client = container_client.get_blob_client(blob_name)

        blob_client.upload_blob(
            json.dumps(log_data, indent=2),
            overwrite=False,
        )

    except Exception as error:
        # Logging failure should NOT break MRI prediction.
        print(f"Blob logging failed: {error}")


# ============================================================
# API ROUTES
# ============================================================


@app.get("/")
def root():
    return {
        "message": "Brain Tumor MRI Classifier API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type could not be determined.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        result = predict_image(image)

        response_data = {
            "filename": file.filename,
            **result,
        }

        # ----------------------------------------------------
        # SAVE PREDICTION LOG TO AZURE BLOB
        # ----------------------------------------------------

        save_prediction_log(
            file.filename,
            response_data,
        )

        return response_data

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed.",
        ) from error