from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Load model once at the start
model_path = os.path.join('ai_models', 'smoke_detection_model.h5')

# Try loading the model and handle potential errors
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(path, target_size=(128, 128)):
    """
    Preprocess the image for model prediction: resize, normalize, and expand dims.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image at path: {path}")
    
    # Convert to RGB in case the image is in BGR format (default for OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to the target size
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    
    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)
    
    return img

def detect_smoke(img_path):
    """
    Detect smoke in an image by using the trained model.
    Returns the detection result and confidence score.
    """
    img_array = preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])  # Ensure it's a regular float for JSON compatibility

    # Define result based on confidence threshold
    if confidence > 0.5:
        result = "🔥 Smoke Detected"
        confidence_percentage = confidence * 100
    else:
        result = "✅ No Smoke"
        confidence_percentage = (1 - confidence) * 100

    # Return the result and confidence percentage as a dictionary
    return {
        'result': result,
        'confidence': confidence_percentage
    }

