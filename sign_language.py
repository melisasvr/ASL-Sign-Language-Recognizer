import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Define ASL alphabet labels (A-Z)
labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
num_classes = len(labels)

# Directory to save dataset
DATA_DIR = 'asl_dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Neural Network Model
class ASLClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=26):
        super(ASLClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout to prevent overfitting
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Function to extract hand landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        data = []
        for lm in landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
        return np.array(data), results.multi_hand_landmarks
    return None, None

# Data Augmentation
def augment_landmarks(landmarks, noise_scale=0.01):
    noise = np.random.normal(0, noise_scale, landmarks.shape)
    return landmarks + noise

# Data Collection
def collect_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        print("Troubleshooting:")
        print("- Ensure no other application is using the webcam.")
        print("- Try a different camera index (e.g., 0, 1, or 2) in cv2.VideoCapture.")
        print("- Check if the webcam drivers are installed.")
        return

    print("Data Collection: Press 'c' to capture, 'n' for next letter, 'q' to quit.")
    print("Tips: Vary hand angles, positions, and lighting for better data diversity.")
    print("Ensure clear distinction between signs (e.g., 'B' flat hand, 'X' bent index finger).")
    current_label_idx = 0
    samples_per_letter = 10  # Reduced to 10 samples per letter

    while current_label_idx < len(labels):
        label = labels[current_label_idx]
        print(f"Collecting data for letter: {label}")
        sample_count = 0
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        while sample_count < samples_per_letter:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Trying again...")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            landmarks, hand_landmarks = extract_landmarks(frame)

            display_frame = frame.copy()
            if hand_landmarks is not None:
                mp_drawing.draw_landmarks(display_frame, hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            cv2.putText(
                display_frame,
                f'Letter: {label} | Samples: {sample_count}/{samples_per_letter} | Press c to capture',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow('Data Collection', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and landmarks is not None:
                np.save(os.path.join(label_dir, f'sample_{sample_count}.npy'), landmarks)
                image_path = os.path.join(label_dir, f'sample_{sample_count}.jpg')
                cv2.imwrite(image_path, frame)
                sample_count += 1
                print(f"Captured sample {sample_count} for letter {label}")
                print(f"Saved image to {image_path}")
                cv2.putText(
                    display_frame,
                    f'Captured sample {sample_count}',
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow('Data Collection', display_frame)
                cv2.waitKey(500)
            elif key == ord('n'):
                current_label_idx += 1
                if current_label_idx < len(labels):
                    print(f"Moving to next letter: {labels[current_label_idx]}")
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection completed.")

# Load Dataset
def load_dataset():
    X, y = [], []
    for label_idx, label in enumerate(labels):
        label_dir = os.path.join(DATA_DIR, label)
        if os.path.exists(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.npy'):
                    landmarks = np.load(os.path.join(label_dir, file))
                    X.append(landmarks)
                    y.append(label_idx)
    return np.array(X), np.array(y)

# Train and Evaluate Model
def train_model():
    X, y = load_dataset()
    if len(X) == 0:
        print("No data found. Please collect data first.")
        return None, None

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Initialize model, loss, and optimizer
    model = ASLClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    batch_size = 32
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            targets = y_train_tensor[i:i+batch_size]
            
            # Augment data
            inputs = inputs + torch.FloatTensor(augment_landmarks(inputs.numpy(), noise_scale=0.01))
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                _, val_pred = torch.max(val_outputs, 1)
                accuracy = (val_pred == y_val_tensor).float().mean().item()
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.4f}')

    # Save model and scaler
    torch.save(model.state_dict(), 'asl_model.pth')
    np.save('scaler_mean.npy', scaler.mean_)
    np.save('scaler_scale.npy', scaler.scale_)
    print("Model training completed.")
    return model, scaler

# Real-Time Inference with Debugging
def real_time_inference(model, scaler):
    model.eval()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        print("Troubleshooting:")
        print("- Ensure no other application is using the webcam.")
        print("- Try a different camera index (e.g., 0, 1, or 2) in cv2.VideoCapture.")
        print("- Check if the webcam drivers are installed.")
        return

    print("Real-Time Inference: Press 'q' to quit, 's' to save frame with prediction.")
    inference_dir = 'inference_debug'
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    frame_count = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Trying again...")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            landmarks, hand_landmarks = extract_landmarks(frame)
            
            predicted_letter = "None"
            confidence = 0.0
            if landmarks is not None:
                # Normalize landmarks
                landmarks = scaler.transform([landmarks])[0]
                landmarks_tensor = torch.FloatTensor(landmarks).unsqueeze(0)
                output = model(landmarks_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_letter = labels[predicted.item()]
                confidence = confidence.item()

                # Draw landmarks
                if hand_landmarks is not None:
                    mp_drawing.draw_landmarks(frame, hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            
            # Display prediction and confidence
            cv2.putText(
                frame,
                f'Predicted: {predicted_letter} ({confidence:.2f})',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow('ASL Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and predicted_letter != "None":
                # Save frame with prediction for debugging
                image_path = os.path.join(inference_dir, f'frame_{frame_count}_{predicted_letter}.jpg')
                cv2.imwrite(image_path, frame)
                print(f"Saved inference frame to {image_path}")
                frame_count += 1
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    print("1. Collect Data")
    print("2. Train Model")
    print("3. Real-Time Inference")
    choice = input("Enter choice (1-3): ")

    if choice == '1':
        collect_data()
    elif choice == '2':
        model, scaler = train_model()
    elif choice == '3':
        try:
            model = ASLClassifier()
            model.load_state_dict(torch.load('asl_model.pth'))
            scaler = StandardScaler()
            scaler.mean_ = np.load('scaler_mean.npy')
            scaler.scale_ = np.load('scaler_scale.npy')
            real_time_inference(model, scaler)
        except FileNotFoundError:
            print("Error: Model files not found. Please train the model first (option 2).")
    else:
        print("Invalid choice.")

    hands.close()