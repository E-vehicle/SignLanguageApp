import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import cv2
import numpy as np
import mediapipe as mp


import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Define custom colors for each finger (as observed from your image)
FINGER_COLORS = {
    'thumb': (176, 230, 255),       # Red
    'index': (122, 60, 126),       # Blue
    'middle': (0, 203, 250),      # Green
    'ring': (48, 255, 48),      # Yellow
    'pinky': (190, 97, 22),     # Purple
    'palm': (127, 127, 127),    # Light gray
}

# Define finger joint connections
FINGER_CONNECTIONS = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
    'palm': [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0)],  # connections around palm
}

# Start webcam
cap = cv2.VideoCapture(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Linear(128, 36)
)
model.load_state_dict(torch.load('sign_language_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Label mapping
idx_to_label = list("0123456789abcdefghijklmnopqrstuvwxyz")  # 26 letters and 10 numbers

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert the frame
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # Create a black canvas
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # do all the mediapipe stuff since this is what we are sending as an input
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            if handedness.classification[0].label != "Right":
                continue

            h, w, _ = frame.shape
            landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Draw fingers
            for finger, indices in FINGER_CONNECTIONS.items():
                color = FINGER_COLORS[finger]
                if finger == 'palm':
                    for start, end in indices:
                        pt1 = landmark_points[start]
                        pt2 = landmark_points[end]
                        cv2.line(black_frame, pt1, pt2, color, 3)
                else:
                    for i in range(len(indices) - 1):
                        pt1 = landmark_points[indices[i]]
                        pt2 = landmark_points[indices[i + 1]]
                        cv2.line(black_frame, pt1, pt2, color, 2)
                    for idx in indices:
                        cv2.circle(black_frame, landmark_points[idx], 5, color, -1)
            
            # Draw red palm circles
            palm_indices = set(i for conn in FINGER_CONNECTIONS['palm'] for i in conn)
            for idx in palm_indices:
                pt = landmark_points[idx]
                cv2.circle(black_frame, pt, 5, (53, 51, 255), -1) # make the palm points red instead of gray so they match our dataset's images

            # Draw wrist base point
            cv2.circle(black_frame, landmark_points[0], 5, FINGER_COLORS['palm'], -1)

    flipped_frame = cv2.flip(black_frame, 1)
    
    # Preprocess frame
    img = transform(flipped_frame)
    img = img.unsqueeze(0).to(device)  # Add batch dimension

    # Prediction using the flipped frame since it looks like in the images, the thumb was on the right side instead of the left side even though sign language uses the right hand
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
        cv2.putText(flipped_frame, text, (10, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw certainty bar
        bar_length = int(confidence * 2)  # Scale bar length
        bar_color = (0, 255, 0) if confidence > 60 else (0, 0, 255)  # Green if >60%, else Red
        cv2.rectangle(flipped_frame, (10, y_start + 10), (10 + bar_length, y_start + 30), bar_color, -1)
        cv2.rectangle(flipped_frame, (10, y_start + 10), (210, y_start + 30), (255, 255, 255), 2)  # Outline

        y_start += 60  # Move down for next prediction
    cv2.imshow('Sign Language Interpreter', flipped_frame) # currently displaying the mediapipe image that is what we feed to the model, this is being displayed for debugging purposes to see what the inputs look like
    # cv2.imshow("Hand Keypoints", black_frame)

    if cv2.waitKey(1) & 0xFF == 'q':
        break

cap.release()
cv2.destroyAllWindows()





# # Load the model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 128),
#     nn.ReLU(),
#     nn.Linear(128, 24)
# )
# model.load_state_dict(torch.load('sign_language_model.pth', map_location=device))
# model = model.to(device)
# model.eval()

# # Label mapping
# idx_to_label = list("abcdefghiklmnopqrstuvwxy")  # 24 letters

# # Preprocessing
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# # Open webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip frame horizontally
#     frame = cv2.flip(frame, 1)

#     # Preprocess frame
#     img = transform(frame)
#     img = img.unsqueeze(0).to(device)  # Add batch dimension

#     # Prediction
#     with torch.no_grad():
#         outputs = model(img)
#         probabilities = torch.softmax(outputs, dim=1)
#         top_probs, top_idxs = torch.topk(probabilities, 3, dim=1)  # Get top 3 predictions

#     top_probs = top_probs.squeeze(0).cpu().numpy()
#     top_idxs = top_idxs.squeeze(0).cpu().numpy()

#     # Display top predictions
#     y_start = 50
#     for i in range(3):
#         label = idx_to_label[top_idxs[i]]
#         confidence = top_probs[i] * 100  # Convert to percentage

#         text = f'{i+1}: {label} ({confidence:.1f}%)'
#         cv2.putText(frame, text, (10, y_start),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

#         # Draw certainty bar
#         bar_length = int(confidence * 2)  # Scale bar length
#         bar_color = (0, 255, 0) if confidence > 60 else (0, 0, 255)  # Green if >60%, else Red
#         cv2.rectangle(frame, (10, y_start + 10), (10 + bar_length, y_start + 30), bar_color, -1)
#         cv2.rectangle(frame, (10, y_start + 10), (210, y_start + 30), (255, 255, 255), 2)  # Outline

#         y_start += 60  # Move down for next prediction

#     cv2.imshow('Sign Language Interpreter', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
