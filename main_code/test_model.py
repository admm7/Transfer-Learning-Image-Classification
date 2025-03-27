import torch
from torchvision import transforms
from PIL import Image
import os

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle complet (puisqu'on l'a enregistré avec torch.save(model))
model = torch.load('data_model/final_model_ft.pth', map_location=DEVICE, weights_only=False)
model.eval()

# Classes
CLASS_NAMES = ['elephant', 'gorilla', 'leopard', 'lion', 'panda', 'rhinoceros']

# Transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prédiction
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
    return CLASS_NAMES[preds.item()]

# Test
if __name__ == '__main__':
    test_images = [
        'image_test/gorilla.jpg',
        'image_test/leopard.jpg',
        'image_test/lion.jpg',
        'image_test/panda.jpg',
        'image_test/rhinoceros.jpg'
    ]
    for img_path in test_images:
        try:
            print(f"{img_path} -> Predicted: {predict(img_path)}")
        except Exception as e:
            print(f"Erreur avec {img_path} : {e}")
