import os
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import io

app = Flask(__name__)
CORS(app)

# Global variables
model = None
device = None
transform = None
face_cascade = None

def load_model():
    global model, device, transform, face_cascade
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Load model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
    
    model_path = "models/antispoof_vit.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("Warning: Model file not found. Using untrained model.")
        print("Please train the model first using train_model.py")
    
    model.eval().to(device)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def crop_face(image):
    """Crop face from image using OpenCV"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

def predict_image(image_pil):
    """Predict if image is real or fake"""
    global model, device, transform
    
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Convert PIL to OpenCV format for face detection
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Try to crop face
        face = crop_face(image_cv)
        if face is None:
            return {"prediction": "No face detected"}
        # Convert back to PIL
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(face_rgb)
        
        # Apply transforms
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).item()
        
        result = "Real" if pred == 0 else "Fake"
        
        return {
            "prediction": result
        }
    
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.json
        image_data = data['image']
        
        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = predict_image(image_pil)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Load model
    load_model()
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
