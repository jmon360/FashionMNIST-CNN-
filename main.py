import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # Needed for visualization

# ───────────────────────────────────────────────
# [1] Load and Normalize the Fashion MNIST Dataset
# ───────────────────────────────────────────────
# Purpose: Prepare training and testing datasets for model ingestion.
# Step 1: Apply tensor conversion and normalize pixel values to [-1, 1].
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Step 2: Download FashionMNIST datasets if not already present.
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# ───────────────────────────────────────────────
# [2] Define DataLoaders
# ───────────────────────────────────────────────
# Purpose: Create iterable batches of data for training and testing.
def get_dataloaders(batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ───────────────────────────────────────────────
# [3] Define the CNN Architecture
# ───────────────────────────────────────────────
# Purpose: Build a convolutional neural network with variable filters.
class FashionCNN(nn.Module):
    def __init__(self, num_filters):
        super(FashionCNN, self).__init__()
        # First conv layer: input channels = 1 (grayscale), output = num_filters
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=5)
        # Second conv layer: doubles output channels
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=5)
        # Fully connected layers after flattening
        self.fc1 = nn.Linear(num_filters * 2 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output layer: 10 clothing classes
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply conv → ReLU → pooling sequentially
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten before entering fully connected layers
        x = x.view(-1, self.num_flat_features(x))
        # Feed through FC layers with ReLU activations
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax needed; CrossEntropyLoss handles that
        return x

    def num_flat_features(self, x):
        # Calculate product of feature map dimensions (excluding batch size)
        size = x.size()[1:]
        return torch.prod(torch.tensor(size)).item()

# ───────────────────────────────────────────────
# [4] Train and Evaluate the Model
# ───────────────────────────────────────────────
# Purpose: Train the CNN for a given hyperparameter setting and report accuracy.
def train_and_evaluate(num_filters, batch_size, epochs=3):
    train_loader, test_loader = get_dataloaders(batch_size)
    model = FashionCNN(num_filters).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # ───── Training Phase ─────
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()              # Reset gradients
            outputs = model(images)            # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()                    # Backpropagation
            optimizer.step()                   # Update weights
            running_loss += loss.item()        # Track loss for reporting
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # ───── Evaluation Phase ─────
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy with Filters: {num_filters}, Batch Size: {batch_size}: {accuracy:.2f}%")

    # Save model checkpoint for reproducibility or future ensemble
    torch.save(model.state_dict(), f"model_filters_{num_filters}_batch_{batch_size}.pth")
    return accuracy

# ───────────────────────────────────────────────
# [5] Hyperparameter Tuning Analysis
# ───────────────────────────────────────────────
# Purpose: Test combinations of filters and batch sizes to find optimal performance.
def hyperparameter_analysis():
    filters_list = [8, 16, 32]
    batch_sizes = [32, 64, 128]
    results = []

    for num_filters in filters_list:
        for batch_size in batch_sizes:
            print(f"Training with Filters: {num_filters}, Batch Size: {batch_size}")
            accuracy = train_and_evaluate(num_filters, batch_size)
            results.append((num_filters, batch_size, accuracy))

    return results

# ───────────────────────────────────────────────
# [6] Visualize Results for Interpretation
# ───────────────────────────────────────────────
# Purpose: Plot how accuracy varies with batch size and number of filters.
def visualize_results(results):
    import pandas as pd

    df = pd.DataFrame(results, columns=["Filters", "Batch Size", "Accuracy"])

    # Accuracy vs Batch Size (for each filter count)
    plt.figure(figsize=(10, 6))
    for filters in df["Filters"].unique():
        subset = df[df["Filters"] == filters]
        plt.plot(subset["Batch Size"], subset["Accuracy"], marker="o", label=f"{filters} Filters")
    plt.title("Accuracy vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy vs Filters (for a selected batch size)
    batch_size_to_plot = 64
    subset = df[df["Batch Size"] == batch_size_to_plot]

    plt.figure(figsize=(10, 6))
    plt.plot(subset["Filters"], subset["Accuracy"], marker="o", label=f"Batch Size {batch_size_to_plot}")
    plt.title("Accuracy vs Number of Filters")
    plt.xlabel("Number of Filters")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ───────────────────────────────────────────────
# [7] Main Execution Block
# ───────────────────────────────────────────────
# Purpose: Run the full pipeline: training, evaluation, visualization.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = hyperparameter_analysis()
    for result in results:
        print(f"Filters: {result[0]}, Batch Size: {result[1]}, Accuracy: {result[2]:.2f}%")
    visualize_results(results)

