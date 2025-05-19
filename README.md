# ASL Sign Language Recognizer
A real-time American Sign Language (ASL) recognition system that uses computer vision to detect and classify hand gestures corresponding to ASL alphabet letters (A-Z).

## Features
1. **Data Collection**: Capture and save hand poses for each letter of the ASL alphabet
2. **Image Saving**: Automatically saves images of your hand signs during data collection
3. **Machine Learning Model**: Trains a neural network to recognize hand poses
4. **Real-Time Recognition**: Processes webcam feed to recognize ASL signs in real-time

## Requirements
- Python 3.7+
- OpenCV
- MediaPipe
- PyTorch
- NumPy
- scikit-learn

## Installation
1. Clone or download this repository
2. Install the required packages:
`pip install opencv-python mediapipe torch numpy scikit-learn`

## Usage
1. Run the script using Python:
`python sign_language.py`
2. **Main Menu Options**:
- When you run the program, you'll see the following options:
1. **Collect Data**: Capture hand poses for training the model
2. **Train Model**: Train the neural network using collected data
3. **Real-Time Inference**: Use the trained model to recognize signs in real-time
- **Data Collection Mode**:
- This mode allows you to collect training data for each letter of the ASL alphabet:
- Controls:
- Press `c `to capture a sample when your hand is in the correct position
- Press `n` to move to the next letter
- Press `q` to quit data collection

- What happens when you capture a sample:
- The hand landmarks (coordinates) are saved as .npy files
- An image of your hand sign is saved as a .jpg file in the same folder
- You'll see a confirmation message showing where the image was saved

- ## Project Structure
```
asl-sign-recognizer/
├── sign_recognizer.py      # Main script for data collection, training, and inference
├── asl_dataset/            # Folder storing training data (landmarks and images)
│   ├── A/                  # Images and landmarks for letter A
│   ├── B/                  # Images and landmarks for letter B
│   └── ...                 # Folders for other letters
├── inference_debug/        # Saved frames from real-time inference for debugging
├── asl_model.pth           # Trained model weights
├── scaler_mean.npy         # Scaler mean for data normalization
└── scaler_scale.npy        # Scaler scale for data normalization
```

## Troubleshooting
1. Webcam Not Working:
- Ensure no other application is using the webcam.
- Try different camera indices (e.g., cv2.VideoCapture(1) or cv2.VideoCapture(2)).
- Check webcam drivers and test with the script above.
2. Incorrect Predictions (e.g., 'B' predicted as 'X', 'C' as 'O'):
- Check Training Data:
- Open asl_dataset/B/ and asl_dataset/X/ images to ensure distinct gestures (e.g., 'B' flat, 'X' bent index finger).
- Collect more diverse data by varying hand angles and lighting.
3. Debug Inference:
- Press s during inference to save frames in inference_debug. Compare with training images.
- Check confidence scores; low scores (<0.7) indicate uncertainty, suggesting more training data is needed.
4. Validation Accuracy:
- If validation accuracy (printed during training) is below 80%, recollect data with clearer gestures or increase samples (e.g., 15 per letter).
5. MediaPipe Warning (landmark_projection_calculator.cc:186):
- This warning is benign and does not significantly affect performance. It occurs due to MediaPipe’s default ROI settings.
- Ensure good lighting and keep your hand fully visible to improve landmark detection.
6. Low Model Accuracy:
- Increase samples per letter (e.g., to 15 or 20) in collect_data by changing samples_per_letter.
- Delete the asl_dataset folder and recollect data with more variation.
- Adjust hidden_size in ASLClassifier to 256 or add more layers if needed.

## Acknowledgments
- Built with MediaPipe for hand landmark detection and PyTorch for neural network training.
- Inspired by ASL recognition tutorials and open-source computer vision projects.

## License
- This project is licensed under the MIT License.


