from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


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
# Image preprocessing
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
# Load models once
# ---------------------------------------------------------

print("Loading binary model...")
binary_model = create_binary_model()
print("Binary model loaded successfully.")

print("Loading multiclass model...")
multiclass_model = create_multiclass_model()
print("Multiclass model loaded successfully.")


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def predict_image(image: Image.Image):
    """
    Run binary and multiclass predictions on one MRI image.
    """

    image = image.convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        # ---------------------------------------------
        # Binary prediction
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Multiclass prediction
        # ---------------------------------------------

        multiclass_output = multiclass_model(input_tensor)

        multiclass_probabilities = torch.softmax(
            multiclass_output,
            dim=1,
        )

        multiclass_confidence, multiclass_class = torch.max(
            multiclass_probabilities,
            dim=1,
        )

        multiclass_index = multiclass_class.item()
        multiclass_confidence_value = multiclass_confidence.item()

        tumor_type = MULTICLASS_CLASSES[multiclass_index]

    # ---------------------------------------------
    # Final result
    # ---------------------------------------------

    if binary_prediction == "No Tumor":
        final_prediction = "No Tumor"
        final_confidence = binary_confidence_value
    else:
        final_prediction = tumor_type
        final_confidence = multiclass_confidence_value

    return {
        "binary_prediction": binary_prediction,
        "binary_confidence": round(binary_confidence_value * 100, 2),
        "tumor_type": tumor_type,
        "tumor_confidence": round(multiclass_confidence_value * 100, 2),
        "final_prediction": final_prediction,
        "final_confidence": round(final_confidence * 100, 2),
    }