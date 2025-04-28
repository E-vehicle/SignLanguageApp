import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import cv2
import numpy as np

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Linear(128, 24)
)
model.load_state_dict(torch.load('sign_language_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Label mapping
idx_to_label = list("abcdefghiklmnopqrstuvwxy")  # 24 letters

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Preprocess frame
    img = transform(frame)
    img = img.unsqueeze(0).to(device)  # Add batch dimension

    # Prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = idx_to_label[predicted.item()]

    # Display the prediction
    cv2.putText(frame, f'Prediction: {label}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Interpreter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
