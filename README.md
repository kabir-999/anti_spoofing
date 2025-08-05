# Anti-Spoofing Face Detection

A real-time face authenticity verification system using Vision Transformer (ViT) and computer vision techniques.

## Features

- **Real-time webcam capture** for face detection
- **AI-powered analysis** using Vision Transformer model
- **Face detection and cropping** using OpenCV
- **Web-based interface** for easy access
- **Deployment ready** for Render platform

## Project Structure

```
anti_spoofing/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment config
├── render.yaml           # Render service config
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend JavaScript
├── models/               # Trained model storage
│   └── antispoof_vit.pth # Trained model (after training)
└── data/                 # Training data (you need to add this)
    ├── real_video/       # Real face images
    └── attack/           # Fake/attack images
```

## Setup Instructions

### 1. Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset:**
   - Create a `data` folder in the project root
   - Add two subfolders: `real_video` and `attack`
   - Place your training images in the respective folders

3. **Train the model:**
   ```bash
   python train_model.py
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the web interface:**
   Open your browser and go to `http://localhost:5000`

### 2. Deployment on Render

1. **Push to GitHub:**
   - Create a new repository on GitHub
   - Push all files to the repository

2. **Deploy on Render:**
   - Go to [Render.com](https://render.com)
   - Create a new Web Service
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` configuration

3. **Upload trained model:**
   - After training locally, you'll need to upload the `models/antispoof_vit.pth` file
   - You can do this through Render's file upload or by committing it to your repository

## Usage

1. **Start Camera:** Click the "Start Camera" button to access your webcam
2. **Position Face:** Make sure your face is clearly visible in the camera view
3. **Capture & Analyze:** Click "Capture & Analyze" to take a photo and get results
4. **View Results:** The system will show:
   - Prediction (Real or Fake)
   - Confidence percentage
   - Whether a face was detected

## Model Details

- **Architecture:** Vision Transformer (ViT) Tiny
- **Input Size:** 224x224 pixels
- **Classes:** 2 (Real, Fake)
- **Preprocessing:** Face detection and cropping using Haar cascades

## Dataset Requirements

Your dataset should be organized as follows:
- `data/real_video/`: Images of real faces
- `data/attack/`: Images of fake faces (photos of photos, screens, etc.)

Supported image formats: PNG, JPG, JPEG

## API Endpoints

- `GET /`: Web interface
- `POST /predict`: Image analysis endpoint
- `GET /health`: Health check endpoint

## Security Considerations

- The model runs inference on the server side
- Images are processed in memory and not stored
- HTTPS should be enabled in production

## Troubleshooting

1. **Camera not working:** Ensure your browser supports WebRTC and you've granted camera permissions
2. **Model not found:** Make sure to train the model first using `train_model.py`
3. **Poor accuracy:** Ensure your training dataset is balanced and high-quality

## License

This project is for educational and research purposes.
