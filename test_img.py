import torch
from torchvision import transforms
from torch.nn.functional import softmax
from PIL import Image
import sys
from network import Network

# Define the same model architecture as in your training
class CatDogCNN(torch.nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(1)

# Preprocessing transformations (same as training val/test)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # increase from (200, 200)
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def predict_image(image_path, model_path='cat_v_dog_cnn.pth', device='cpu'):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model = Network()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        print("Output shape:", output.shape)

        if output.dim() == 1:
            probabilities = softmax(output, dim=0)
            predicted_class = torch.argmax(probabilities).item()
            prob = probabilities[predicted_class].item()
        else:
            probabilities = softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            prob = probabilities[0][predicted_class].item()

        label = "dog" if predicted_class == 1 else "cat"
        print(f"Prediction: {label} (probability: {prob:.4f})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_single_image.py path/to/image.jpg")
        sys.exit(1)
    img_path = sys.argv[1]
    predict_image(img_path)
