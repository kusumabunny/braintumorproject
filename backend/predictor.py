from pathlib import Path

import torch
from PIL import Image
from torch import nn
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

# Heuristic threshold for the MRI input-validation gate: the share of
# softmax probability (over all prompts) that lands on the MRI prompts.
# On 158 real MRIs the lowest share was 0.65; the highest non-MRI
# (grayscale brain drawing) was 0.47. Not a medically validated value.
CLIP_THRESHOLD = 0.55

# Brain MRIs are grayscale. Mean per-pixel channel difference above this
# value means the upload is a colour image (fruit, cars, people, ...).
# Real MRIs measured 0.0; colour photos measured 11-91.
MAX_COLORFULNESS = 10.0

MRI_PROMPTS = [
    "a grayscale axial brain MRI scan showing the brain",
    "a clinical radiology image of a human brain MRI",
    "a T1 weighted magnetic resonance image of the brain",
    "a T2 weighted magnetic resonance image of the brain",
    "a medical scan of the human brain from an MRI machine",
    "a cross-sectional MRI image of the human brain",
]

NON_MRI_PROMPTS = [
    "a photograph of a banana",
    "a photograph of fruit",
    "a photograph of food",
    "a grayscale photograph of food",
    "a photograph of a person",
    "a photograph of an animal",
    "a photograph of a car",
    "a photograph of a building",
    "a photograph of an everyday object",
    "a natural landscape photograph",
    "a black and white photograph",
    "a screenshot of a computer application",
    "a scanned text document",
    "a medical photograph that is not an MRI scan",
    "a CT scan or X-ray image, not an MRI",
    "a drawing or cartoon",
    "an anatomical illustration or diagram of the brain",
]


# =========================================================
# Load CLIP
# =========================================================

clip_processor = CLIPProcessor.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL_NAME
)

clip_model.to(DEVICE)  # type: ignore[arg-type]  # transformers stub quirk
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

def colorfulness(image: Image.Image) -> float:
    """
    Mean absolute difference between colour channels (0-255 scale).
    Grayscale images such as MRI scans score ~0.
    """

    pixels = transforms.functional.pil_to_tensor(
        image.convert("RGB").resize((224, 224))
    ).float()

    red, green, blue = pixels[0], pixels[1], pixels[2]

    difference = (
        (red - green).abs()
        + (green - blue).abs()
        + (red - blue).abs()
    )

    return difference.mean().item() / 3


def check_if_brain_mri(image: Image.Image):
    """
    Heuristic input-validation gate:

    1. Reject colour images (MRI scans are grayscale).
    2. Use CLIP zero-shot scores against MRI and non-MRI
       prompts; accept when the MRI prompts receive at least
       CLIP_THRESHOLD of the total probability.

    CLIP cannot reliably tell a brain CT from a brain MRI, so this
    gate rejects unrelated images but is not a guaranteed MRI detector.
    """

    image = image.convert("RGB")

    image_colorfulness = colorfulness(image)

    if image_colorfulness > MAX_COLORFULNESS:
        print(
            "[MRI GATE] "
            f"colorfulness={image_colorfulness:.2f} > "
            f"{MAX_COLORFULNESS}, accepted=False"
        )
        return False, 0.0

    text_prompts = MRI_PROMPTS + NON_MRI_PROMPTS

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

    # Softmax over every prompt, then add up the probability that
    # lands on the MRI prompts. Summing over all prompts is far more
    # stable than comparing only the single best prompt per group.
    prompt_probabilities = torch.softmax(logits, dim=0)

    mri_probability = (
        prompt_probabilities[:len(MRI_PROMPTS)].sum().item()
    )

    is_brain_mri = mri_probability >= CLIP_THRESHOLD

    best_non_mri_prompt = NON_MRI_PROMPTS[
        int(prompt_probabilities[len(MRI_PROMPTS):].argmax().item())
    ]

    print(
        "[MRI GATE] "
        f"colorfulness={image_colorfulness:.2f}, "
        f"MRI share={mri_probability:.4f} "
        f"(threshold {CLIP_THRESHOLD}), "
        f"closest non-MRI prompt='{best_non_mri_prompt}', "
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

    binary_index = int(binary_class.item())
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

    filtered_index = int(tumor_class.item())
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