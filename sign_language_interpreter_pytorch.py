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

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Preprocess frame
    img = transform(frame)
    img = img.unsqueeze(0).to(device)  # Add batch dimension

    # Prediction
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probabilities, 3, dim=1)  # Get top 3 predictions

    top_probs = top_probs.squeeze(0).cpu().numpy()
    top_idxs = top_idxs.squeeze(0).cpu().numpy()

    # Display top predictions
    y_start = 50
    for i in range(3):
        label = idx_to_label[top_idxs[i]]
        confidence = top_probs[i] * 100  # Convert to percentage

        text = f'{i+1}: {label} ({confidence:.1f}%)'
        cv2.putText(frame, text, (10, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw certainty bar
        bar_length = int(confidence * 2)  # Scale bar length
        bar_color = (0, 255, 0) if confidence > 60 else (0, 0, 255)  # Green if >60%, else Red
        cv2.rectangle(frame, (10, y_start + 10), (10 + bar_length, y_start + 30), bar_color, -1)
        cv2.rectangle(frame, (10, y_start + 10), (210, y_start + 30), (255, 255, 255), 2)  # Outline

        y_start += 60  # Move down for next prediction

    cv2.imshow('Sign Language Interpreter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
