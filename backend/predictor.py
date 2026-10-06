from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BINARY_MODEL_PATH = PROJECT_ROOT / "models" / "best_binary_resnet18.pth"

MULTICLASS_MODEL_PATH = (
    PROJECT_ROOT / "models" / "brain_tumor_resnet18_final.pth"
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# Class names
# ---------------------------------------------------------

# Binary model:
# 0 = No Tumor
# 1 = Tumor

BINARY_CLASSES = {
    0: "No Tumor",
    1: "Tumor",
}


# Multiclass model:
# 0 = Glioma
# 1 = Meningioma
# 2 = No Tumor
# 3 = Pituitary

MULTICLASS_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}


# ---------------------------------------------------------
# Image preprocessing for ResNet
# ---------------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ---------------------------------------------------------
# CLIP configuration
# ---------------------------------------------------------

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Minimum CLIP probability required to accept the image
# as a brain MRI.
#
# If the probability is below this value, the image is
# rejected as an unrelated image.
CLIP_THRESHOLD = 0.60


# ---------------------------------------------------------
# Load CLIP
# ---------------------------------------------------------

print("Loading CLIP model...")

clip_processor = CLIPProcessor.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model.to(DEVICE)
clip_model.eval()

print("CLIP model loaded successfully.")


# ---------------------------------------------------------
# Create Binary ResNet18
# ---------------------------------------------------------

def create_binary_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2,
    )

    state_dict = torch.load(
        BINARY_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------
# Create Multiclass ResNet18
# ---------------------------------------------------------

def create_multiclass_model():
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        4,
    )

    state_dict = torch.load(
        MULTICLASS_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------
# Load ResNet models once
# ---------------------------------------------------------

print("Loading binary model...")

binary_model = create_binary_model()

print("Binary model loaded successfully.")


print("Loading multiclass model...")

multiclass_model = create_multiclass_model()

print("Multiclass model loaded successfully.")


# ---------------------------------------------------------
# CLIP MRI validation
# ---------------------------------------------------------

def check_if_brain_mri(image: Image.Image):
    """
    Use CLIP to determine whether the uploaded image
    appears to be a brain MRI or an unrelated image.

    Returns:
        is_brain_mri: bool
        confidence: float
    """

    image = image.convert("RGB")

    text_prompts = [
        "a brain MRI scan",
        "a medical MRI image of the human brain",
        "an MRI scan of a human brain",
        "a brain medical imaging scan",
        "an unrelated image",
        "a photograph that is not a medical image",
        "a non-medical image",
        "an image that is not a brain MRI",
    ]

    inputs = clip_processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = clip_model(**inputs)

        logits = outputs.logits_per_image

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    # First four prompts represent brain MRI.
    brain_mri_probability = probabilities[:4].mean().item()

    # Last four prompts represent unrelated/non-MRI.
    unrelated_probability = probabilities[4:].mean().item()

    total_probability = (
        brain_mri_probability
        + unrelated_probability
    )

    # Normalize the two groups.
    if total_probability > 0:
        brain_mri_score = (
            brain_mri_probability
            / total_probability
        )
    else:
        brain_mri_score = 0.0

    is_brain_mri = (
        brain_mri_score >= CLIP_THRESHOLD
    )

    return (
        is_brain_mri,
        brain_mri_score,
    )


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def predict_image(image: Image.Image):
    """
    Complete prediction pipeline:

    1. CLIP validates whether the image appears to be
       a brain MRI.
    2. Binary ResNet18 predicts Tumor / No Tumor.
    3. If Tumor, multiclass ResNet18 predicts tumor type.
    """

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if image is None:
        raise ValueError("Image cannot be None.")

    image = image.convert("RGB")


    # -----------------------------------------------------
    # STEP 1: CLIP unrelated-image detection
    # -----------------------------------------------------

    is_brain_mri, clip_confidence = check_if_brain_mri(
        image
    )

    print(
        f"CLIP brain MRI confidence: "
        f"{clip_confidence * 100:.2f}%"
    )


    # -----------------------------------------------------
    # Reject unrelated images
    # -----------------------------------------------------

    if not is_brain_mri:

        return {
            "binary_prediction": "Unrelated Image",
            "binary_confidence": round(
                (1 - clip_confidence) * 100,
                2,
            ),
            "tumor_type": "Unrelated Image",
            "tumor_confidence": 0.0,
            "final_prediction": "Unrelated Image",
            "final_confidence": round(
                (1 - clip_confidence) * 100,
                2,
            ),
        }


    # -----------------------------------------------------
    # STEP 2: ResNet preprocessing
    # -----------------------------------------------------

    input_tensor = transform(image)

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    # -----------------------------------------------------
    # STEP 3: Binary prediction
    # -----------------------------------------------------

    with torch.no_grad():

        binary_output = binary_model(
            input_tensor
        )

        binary_probabilities = torch.softmax(
            binary_output,
            dim=1,
        )

        binary_confidence, binary_class = torch.max(
            binary_probabilities,
            dim=1,
        )

        binary_index = binary_class.item()

        binary_confidence_value = (
            binary_confidence.item()
        )

        binary_prediction = BINARY_CLASSES[
            binary_index
        ]


        # -------------------------------------------------
        # STEP 4: Multiclass prediction
        # -------------------------------------------------

        multiclass_output = multiclass_model(
            input_tensor
        )

        multiclass_probabilities = torch.softmax(
            multiclass_output,
            dim=1,
        )

        multiclass_confidence, multiclass_class = torch.max(
            multiclass_probabilities,
            dim=1,
        )

        multiclass_index = multiclass_class.item()

        multiclass_confidence_value = (
            multiclass_confidence.item()
        )

        tumor_type = MULTICLASS_CLASSES[
            multiclass_index
        ]


    # -----------------------------------------------------
    # STEP 5: Final result
    # -----------------------------------------------------

    if binary_prediction == "No Tumor":

        final_prediction = "No Tumor"

        final_confidence = (
            binary_confidence_value
        )

    else:

        final_prediction = tumor_type

        final_confidence = (
            multiclass_confidence_value
        )


    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {
        "binary_prediction": binary_prediction,
        "binary_confidence": round(
            binary_confidence_value * 100,
            2,
        ),
        "tumor_type": tumor_type,
        "tumor_confidence": round(
            multiclass_confidence_value * 100,
            2,
        ),
        "final_prediction": final_prediction,
        "final_confidence": round(
            final_confidence * 100,
            2,
        ),
    }