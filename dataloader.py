import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from resnet import resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# 加载数据集
train_dataset = ImageFolder('./archive/Apple/Train', transform=transform)
test_dataset = ImageFolder('./archive/Apple/Test', transform=transform)

# Create data loaders for the datasets
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Load the pretrained ResNet-50 model
resnet = resnet50()
resnet.eval()

# Function to extract features from the dataloader using ResNet-50
def extract_features(loader, model):
    features = []
    labels = []

    with torch.no_grad():
        for images, target in loader:
            features_batch = model(images)
            features_batch = features_batch.view(features_batch.size(0), -1)
            features.append(features_batch)
            labels.append(target)

    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

# Extract features from the train and test datasets
train_features, train_labels = extract_features(train_loader, resnet)
test_features, test_labels = extract_features(test_loader, resnet)

# Define the logistic regression model
input_size = train_features.shape[1]
output_size = len(train_dataset.classes)
logreg = nn.Linear(input_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(logreg.parameters(), lr=0.001, weight_decay=1)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    for features_batch, labels in train_loader:
        optimizer.zero_grad()
        outputs = logreg(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# Evaluation on the test set
logreg.eval()
with torch.no_grad():
    outputs = logreg(test_features)
    _, predicted = torch.max(outputs, dim=1)

# Convert the tensors to numpy arrays for evaluation
test_labels_np = test_labels.numpy()
predicted_np = predicted.numpy()

# Compute and print accuracy, precision, recall, and F1-score
accuracy = accuracy_score(test_labels_np, predicted_np)
precision = precision_score(test_labels_np, predicted_np, average='weighted')
recall = recall_score(test_labels_np, predicted_np, average='weighted')
f1 = f1_score(test_labels_np, predicted_np, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Compute and print confusion matrix
cm = confusion_matrix(test_labels_np, predicted_np)
print("Confusion Matrix:")
print(cm)

all_probabilities = nn.functional.softmax(outputs, dim=1).numpy()
roc_auc = dict()
plt.figure(figsize=(8, 6))
for i in range(output_size):
    fpr, tpr, _ = roc_curve(test_labels_np == i, all_probabilities[:, i])
    roc_auc[i] = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()