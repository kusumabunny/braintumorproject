import torch


binary_path = "models/best_binary_resnet18.pth"
multiclass_path = "models/brain_tumor_resnet18_final.pth"


print("=" * 60)
print("BINARY MODEL")
print("=" * 60)

binary_model = torch.load(
    binary_path,
    map_location="cpu",
    weights_only=False,
)

print("Type:", type(binary_model))

if isinstance(binary_model, dict):
    print("Dictionary keys:")
    print(binary_model.keys())


print()
print("=" * 60)
print("MULTICLASS MODEL")
print("=" * 60)

multiclass_model = torch.load(
    multiclass_path,
    map_location="cpu",
    weights_only=False,
)

print("Type:", type(multiclass_model))

if isinstance(multiclass_model, dict):
    print("Dictionary keys:")
    print(multiclass_model.keys())


print()
print("=" * 60)
print("MODEL FILE CHECK COMPLETE")
print("=" * 60)