from PIL import Image

from backend.predictor import predict_image


# Create a temporary test image.
# This only checks that the complete prediction pipeline works.
image = Image.new("RGB", (224, 224), "black")


result = predict_image(image)


print()
print("=" * 60)
print("PREDICTION TEST")
print("=" * 60)

for key, value in result.items():
    print(f"{key}: {value}")

print("=" * 60)
print("PREDICTION PIPELINE WORKING")
print("=" * 60)