from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.predictor import predict_image


app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description="FastAPI backend for Brain Tumor MRI classification",
    version="1.0.0",
)


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

        result = predict_image(image)

        return {
            "filename": file.filename,
            **result,
        }

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