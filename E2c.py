#Load libraries
import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.img_labels = torch.zeros(200, dtype=torch.long)
        self.img_labels[:100] = 0
        self.img_labels[100:] = 1
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Grayscale(),
            transforms.Normalize(mean=(0.5), std=(0.5)),
            transforms.Resize((20, 20), antialias=True)])
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        label = self.img_labels[idx]
        if label == 0:
            img_path = os.path.join(self.img_dir, f"Circle_{idx}.png")
        elif label == 1:
            img_path = os.path.join(self.img_dir, f"Square_{idx-100}.png")
        image = torchvision.io.read_image(img_path)
        image = self.transform(image)
        return image, label


dataset = ShapeDataset("C:/Users/TJ/Documents/Repos/AI-Exam/data/combined")
train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, test_size=0.2)

batch_size = 16
learning_rate = 0.001
epochs = 50

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(20 * 20, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
    nn.Softmax()
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Test the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

plt.plot(losses)
plt.title("Cross Entropy Losses over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.grid()
plt.show()