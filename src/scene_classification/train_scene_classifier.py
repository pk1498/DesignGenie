import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_preprocessing.data_loader import load_sun397_dataset
from src.scene_classification.scene_classifier import SceneClassifier


def train_scene_classifier():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    num_classes = 397  # SUN397 has 397 scene categories

    dataloader = load_sun397_dataset(batch_size=batch_size)

    model = SceneClassifier(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {correct/total:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "models/scene_classifier/scene_classifier.pth")
    print("Model saved to 'models/scene_classifier/scene_classifier.pth'")
