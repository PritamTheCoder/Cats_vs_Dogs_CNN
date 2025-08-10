from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets.folder import DatasetFolder
from network import Network

# Transformations
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load full dataset
dataset = ImageFolder("data/train")

print("Classes found:", dataset.classes)
print("Class to idx:", dataset.class_to_idx)

allowed_classes = ['cats', 'dogs']  # match your folder names exactly
allowed_class_indices = [dataset.class_to_idx[c] for c in allowed_classes]

# Filter dataset samples to only allowed classes
filtered_samples = [s for s in dataset.samples if s[1] in allowed_class_indices]

# Custom filtered dataset without scanning directories again
class FilteredDataset(DatasetFolder):
    def __init__(self, samples, transform=None):
        self.loader = dataset.loader
        self.extensions = dataset.extensions
        self.samples = samples
        self.transform = transform

        self.targets = [allowed_class_indices.index(s[1]) for s in samples]
        self.classes = allowed_classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        sample = self.loader(path)
        if self.transform:
            sample = self.transform(sample)
        target = self.targets[idx]
        return sample, target

    def __len__(self):
        return len(self.samples)

filtered_dataset = FilteredDataset(filtered_samples, transform)

# Split indices with stratification
indices = list(range(len(filtered_dataset)))
train_indices, test_indices = train_test_split(
    indices, test_size=0.2, random_state=0, stratify=filtered_dataset.targets
)

# Subsets and dataloaders
train_subset = Subset(filtered_dataset, train_indices)
test_subset = Subset(filtered_dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=10, shuffle=False)

# Model, loss, optimizer
network = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)

# Training loop
epochs = 8
for epoch in range(epochs):
    network.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = network(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
network.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = network(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save model
torch.save(network.state_dict(), "cat_v_dog_cnn.pth")
print("Model saved as cat_v_dog_cnn.pth")
