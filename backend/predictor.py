from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BINARY_MODEL_PATH = (
    PROJECT_ROOT / "models" / "best_binary_resnet18.pth"
)

MULTICLASS_MODEL_PATH = (
    PROJECT_ROOT / "models" / "brain_tumor_resnet18_final.pth"
)


# =========================================================
# Device
# =========================================================

DEVICE = torch.device("cpu")


# =========================================================
# Class names
# =========================================================

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


# =========================================================
# Image preprocessing for ResNet
# =========================================================

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


# =========================================================
# CLIP configuration
# =========================================================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Heuristic threshold for the MRI input-validation gate.
# This is not a medically validated probability threshold.
CLIP_THRESHOLD = 0.80


# =========================================================
# Load CLIP
# =========================================================

clip_processor = CLIPProcessor.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model.to(DEVICE)
clip_model.eval()


# =========================================================
# Create Binary ResNet18
# =========================================================

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


# =========================================================
# Create Multiclass ResNet18
# =========================================================

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


# =========================================================
# Load ResNet models once
# =========================================================

binary_model = create_binary_model()
multiclass_model = create_multiclass_model()


# =========================================================
# CLIP MRI validation
# =========================================================

def check_if_brain_mri(image: Image.Image):
    """
    Use CLIP as a heuristic input-validation gate.

    CLIP scores are image-text similarity scores, not
    calibrated probabilities. This gate can reject obvious
    unrelated images, but it is not a guaranteed MRI detector.
    """

    image = image.convert("RGB")

    mri_prompts = [
        "a grayscale axial brain MRI scan showing the brain",
        "a clinical radiology image of a human brain MRI",
        "a T1 weighted magnetic resonance image of the brain",
        "a T2 weighted magnetic resonance image of the brain",
        "a medical scan of the human brain from an MRI machine",
        "a cross-sectional MRI image of the human brain",
    ]

    non_mri_prompts = [
        "a photograph of a banana",
        "a photograph of fruit",
        "a photograph of a person",
        "a photograph of an animal",
        "a photograph of a car",
        "a photograph of a building",
        "a photograph of an everyday object",
        "a natural landscape photograph",
        "a screenshot of a computer application",
        "a medical photograph that is not an MRI scan",
        "a CT scan or X-ray image, not an MRI",
        "a drawing or cartoon",
    ]

    text_prompts = mri_prompts + non_mri_prompts

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
        logits = outputs.logits_per_image[0]

    # Separate MRI and non-MRI image-text scores.
    mri_logits = logits[:len(mri_prompts)]
    non_mri_logits = logits[len(mri_prompts):]

    # Compare the strongest matching prompt in each group.
    best_mri_score = mri_logits.max()
    best_non_mri_score = non_mri_logits.max()

    # Relative two-group score. This is not a calibrated
    # probability that the image is actually a brain MRI.
    group_probabilities = torch.softmax(
        torch.stack(
            [
                best_mri_score,
                best_non_mri_score,
            ]
        ),
        dim=0,
    )

    mri_probability = group_probabilities[0].item()
    non_mri_probability = group_probabilities[1].item()

    is_brain_mri = (
        mri_probability >= CLIP_THRESHOLD
        and best_mri_score.item() > best_non_mri_score.item()
    )

    print(
        "[CLIP DEBUG] "
        f"MRI score={best_mri_score.item():.4f}, "
        f"non-MRI score={best_non_mri_score.item():.4f}, "
        f"MRI relative score={mri_probability:.4f}, "
        f"non-MRI relative score={non_mri_probability:.4f}, "
        f"accepted={is_brain_mri}"
    )

    return is_brain_mri, mri_probability


# =========================================================
# Main prediction function
# =========================================================

def predict_image(image: Image.Image):
    """
    Prediction pipeline:

        Uploaded image
              |
              v
        CLIP input validation
              |
        +-----+-----+
        |           |
      Reject       Accept
        |           |
        v           v
      Unrelated   Binary ResNet18
                      |
                +-----+-----+
                |           |
             No Tumor      Tumor
                |           |
                v           v
             No Tumor   Multiclass ResNet18
                            |
                     Glioma / Meningioma /
                           Pituitary

    This is an experimental ML project, not a medical
    diagnostic system.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    image = image.convert("RGB")

    # =====================================================
    # STEP 1: Validate input with CLIP
    # =====================================================

    is_brain_mri, _ = check_if_brain_mri(image)

    # =====================================================
    # STEP 2: Reject images that fail the validation gate
    # =====================================================

    if not is_brain_mri:
        print("[PREDICTION] Rejected by CLIP input validation.")

        return {
            "binary_prediction": "Unrelated Image",
            "binary_confidence": 0.0,
            "tumor_type": "Unrelated Image",
            "tumor_confidence": 0.0,
            "final_prediction": "Unrelated Image",
            "final_confidence": 0.0,
        }

    # =====================================================
    # STEP 3: Preprocess the accepted image
    # =====================================================

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    # =====================================================
    # STEP 4: Binary ResNet18 prediction
    # =====================================================

    with torch.no_grad():
        binary_output = binary_model(input_tensor)

        binary_probabilities = torch.softmax(
            binary_output,
            dim=1,
        )

        binary_confidence, binary_class = torch.max(
            binary_probabilities,
            dim=1,
        )

    binary_index = binary_class.item()
    binary_confidence_value = binary_confidence.item()

    binary_prediction = BINARY_CLASSES[binary_index]

    print(
        f"[BINARY MODEL] Prediction={binary_prediction}, "
        f"confidence={binary_confidence_value:.4f}"
    )

    # =====================================================
    # STEP 5: If No Tumor, stop here
    # =====================================================

    if binary_prediction == "No Tumor":
        confidence_percent = round(
            binary_confidence_value * 100,
            2,
        )

        return {
            "binary_prediction": "No Tumor",
            "binary_confidence": confidence_percent,
            "tumor_type": "No Tumor",
            "tumor_confidence": 0.0,
            "final_prediction": "No Tumor",
            "final_confidence": confidence_percent,
        }

    # =====================================================
    # STEP 6: Multiclass tumor classification
    # =====================================================

    # Original class mapping:
    # 0 = Glioma
    # 1 = Meningioma
    # 2 = No Tumor
    # 3 = Pituitary
    #
    # Since the binary model predicted Tumor, exclude
    # the No Tumor output from the tumor-type decision.

    with torch.no_grad():
        multiclass_output = multiclass_model(input_tensor)
        multiclass_logits = multiclass_output[0]

        tumor_logits = torch.stack(
            [
                multiclass_logits[0],  # Glioma
                multiclass_logits[1],  # Meningioma
                multiclass_logits[3],  # Pituitary
            ]
        )

        tumor_probabilities = torch.softmax(
            tumor_logits,
            dim=0,
        )

        tumor_confidence, tumor_class = torch.max(
            tumor_probabilities,
            dim=0,
        )

    filtered_index = tumor_class.item()
    tumor_confidence_value = tumor_confidence.item()

    # Map filtered indices back to original class indices.
    tumor_class_mapping = {
        0: 0,  # Glioma
        1: 1,  # Meningioma
        2: 3,  # Pituitary
    }

    original_class_index = tumor_class_mapping[filtered_index]
    tumor_type = MULTICLASS_CLASSES[original_class_index]

    print(
        f"[MULTICLASS MODEL] Prediction={tumor_type}, "
        f"confidence={tumor_confidence_value:.4f}"
    )

    # =====================================================
    # STEP 7: Return prediction
    # =====================================================

    return {
        "binary_prediction": "Tumor",
        "binary_confidence": round(
            binary_confidence_value * 100,
            2,
        ),
        "tumor_type": tumor_type,
        "tumor_confidence": round(
            tumor_confidence_value * 100,
            2,
        ),
        "final_prediction": tumor_type,
        "final_confidence": round(
            tumor_confidence_value * 100,
            2,
        ),
    }