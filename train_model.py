import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import timm
import torch.nn as nn
import torch.optim as optim

# Initialize face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Check if root_dir exists
        if not os.path.exists(root_dir):
            print(f"Warning: Dataset directory {root_dir} not found!")
            print("Please ensure your dataset is in the 'frames_anti_spoof_dataset' folder")
            print("Expected structure:")
            print("frames_anti_spoof_dataset/")
            print("  ├── real_video/")
            print("  └── attack/")
            return

        for label, cls in enumerate(['real_video', 'attack']):
            cls_folder = os.path.join(root_dir, cls)
            if not os.path.exists(cls_folder):
                print(f"Warning: {cls_folder} not found!")
                continue
                
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images")

    def crop_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        return image[y:y+h, x:x+w]

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = cv2.imread(path)
            if image is None:
                print(f"Warning: Could not load image {path}")
                # Return a dummy image
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            face = self.crop_face(image)
            if face is None:
                face = image  # fallback

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)

            if self.transform:
                face = self.transform(face)

            return face, label
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return dummy data
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label

    def __len__(self):
        return len(self.image_paths)

def train_model():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Create dataset
    dataset = FaceDataset("frames_anti_spoof_dataset", transform=transform)
    
    if len(dataset) == 0:
        print("No data found! Please add your dataset to the 'data' folder.")
        return

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model setup
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 2)
    model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Training loop
    print("Starting training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")

        acc = correct / total_samples
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.4f}")

    # Evaluation
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "models/antispoof_vit.pth")
    print("Model saved to models/antispoof_vit.pth")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    
    train_model()
