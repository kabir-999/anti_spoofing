class AntiSpoofingApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        
        this.startCameraBtn = document.getElementById('startCamera');
        this.captureBtn = document.getElementById('captureBtn');
        this.stopCameraBtn = document.getElementById('stopCamera');
        
        this.loading = document.getElementById('loading');
        this.results = document.getElementById('results');
        this.error = document.getElementById('error');
        
        this.initEventListeners();
    }
    
    initEventListeners() {
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.captureBtn.addEventListener('click', () => this.captureAndAnalyze());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
    }
    
    async startCamera() {
        try {
            this.hideResults();
            
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            };
            
            this.startCameraBtn.disabled = true;
            this.captureBtn.disabled = false;
            this.stopCameraBtn.disabled = false;
            
        } catch (error) {
            this.showError('Failed to access camera: ' + error.message);
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.video.srcObject = null;
        
        this.startCameraBtn.disabled = false;
        this.captureBtn.disabled = true;
        this.stopCameraBtn.disabled = true;
        
        this.hideResults();
    }
    
    captureImage() {
        if (!this.stream) {
            throw new Error('Camera not started');
        }
        
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        return this.canvas.toDataURL('image/jpeg', 0.8);
    }
    
    async captureAndAnalyze() {
        try {
            this.hideResults();
            this.showLoading();
            
            const imageData = this.captureImage();
            const result = await this.analyzeImage(imageData);
            
            this.hideLoading();
            this.showResults(result);
            
        } catch (error) {
            this.hideLoading();
            this.showError('Analysis failed: ' + error.message);
        }
    }
    
    async analyzeImage(imageData) {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        return result;
    }
    
    showLoading() {
        this.loading.style.display = 'block';
    }
    
    hideLoading() {
        this.loading.style.display = 'none';
    }
    
    showResults(result) {
        const predictionText = document.getElementById('predictionText');
        predictionText.textContent = result.prediction;
        predictionText.className = 'value ' + (result.prediction === 'Real' ? 'real' : 'fake');
        this.results.style.display = 'block';
    }
    
    showError(message) {
        const errorText = document.getElementById('errorText');
        errorText.textContent = message;
        this.error.style.display = 'block';
    }
    
    hideResults() {
        this.results.style.display = 'none';
        this.error.style.display = 'none';
    }
}

// Camera permission modal logic
function showCameraModal() {
    const modal = document.getElementById('cameraModal');
    if (modal) modal.style.display = 'flex';
}
function hideCameraModal() {
    const modal = document.getElementById('cameraModal');
    if (modal) modal.style.display = 'none';
}

let appInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    appInstance = new AntiSpoofingApp();

    // Only show modal if browser supports camera
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        showCameraModal();
        document.getElementById('modalAllowBtn').onclick = async () => {
            hideCameraModal();
            try {
                await appInstance.startCamera();
            } catch (e) {
                // If permission denied, show modal again
                showCameraModal();
            }
        };
    } else {
        // Fallback for unsupported browsers
        const errorDiv = document.getElementById('error');
        const errorText = document.getElementById('errorText');
        if (errorDiv && errorText) {
            errorText.textContent = 'Your browser does not support camera access. Please use a modern browser like Chrome, Firefox, or Safari.';
            errorDiv.style.display = 'block';
        }
    }
});

