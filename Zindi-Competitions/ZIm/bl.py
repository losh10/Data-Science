
import os

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix,classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms


if torch.cuda.is_available():
    device = "cuda"  # Use NVIDIA GPU if available
elif torch.backends.mps.is_available():
    device = "mps"   # Use Apple GPU if available
else:
    device = "cpu"   # Default to CPU if no accelerators found


train_data_dir = os.path.join("crop pictures", "train")  # Path to training data
val_data_dir = os.path.join("crop pictures", "val")    # Path to val data

# Print the paths for verification and debugging purposes
print("Training Data Directory:", train_data_dir)  # Show training data path
print("val Data Directory:", val_data_dir)       # Show val data path

classes = os.listdir(train_data_dir)

print("List of classes:", classes)


train_counts = {}
val_counts = {}

for class_name in classes:
    train_class_dir = os.path.join(train_data_dir, class_name)
    val_class_dir = os.path.join(val_data_dir, class_name)
    
    train_counts[class_name] = len(os.listdir(train_class_dir))
    val_counts[class_name] = len(os.listdir(val_class_dir))

print("Training set counts:")
for class_name, count in train_counts.items():
    print(f"{class_name}: {count} images")

print("\nValidation set counts:") 
for class_name, count in val_counts.items():
    print(f"{class_name}: {count} images")


def convert_to_rgb(img):
    """Convert PIL image to RGB format if it isn't already.
    
    Args:
        img: PIL Image object
    
    Returns:
        PIL Image object in RGB format
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

transform = transforms.Compose([
    transforms.Lambda(convert_to_rgb),  # First convert to RGB if needed
    transforms.Resize((224, 224)),      # Resize to 224x224
    transforms.ToTensor(),              # Convert to tensor
])


print(type(transform))
print("-----------------")
print(transform)

# The transform will be applied to each image (conversion to RGB, resizing, and tensor conversion)
training_dataset = datasets.ImageFolder(train_data_dir, transform)  # Training data with transformations
val_dataset = datasets.ImageFolder(val_data_dir, transform)        # Validation data with same transformations

# Print dataset sizes to verify data loading and ensure proper train/val split
print('Length of training dataset:', len(training_dataset))  # Shows number of training samples
print('Length of validation dataset:', len(val_dataset))     # Shows number of validation samples

def class_counts(dataset):
    """Counts the number of samples per class in a dataset.
    
    Args:
        dataset: A PyTorch Dataset object with class labels
        
    Returns:
        A pandas Series with class names as index and counts as values
    """
    counts = {}
    for _, label in dataset:
        class_name = dataset.classes[label]  # Get class name from label index
        counts[class_name] = counts.get(class_name, 0) + 1  # Increment count
    return pd.Series(counts)

# Calculate and display class counts for both datasets
print("Computing class counts for training data...")
train_counts = class_counts(training_dataset)
print("Training data counts:")
print(train_counts)
print("\nComputing class counts for val data...")
val_counts = class_counts(val_dataset)
print("Validation data counts:")
print(val_counts)

# Create visualization comparing class distributions
plt.figure(figsize=(18, 6))  # Set figure size

# Training data distribution plot
plt.subplot(1, 2, 1)  # First subplot
ax1 = train_counts.sort_values().plot(
    kind='bar',
    color='lightgreen',
    edgecolor='black',
    width=0.8
)
# Add count labels above bars
for p in ax1.patches:
    ax1.annotate(str(p.get_height()), 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 5), 
                textcoords='offset points')
plt.xlabel("Class Label", fontsize=12)
plt.ylabel("Frequency [count]", fontsize=12)
plt.title("Training Dataset Class Distribution", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.grid(axis='y', alpha=0.3)  # Add light grid lines

# Validation data distribution plot
plt.subplot(1, 2, 2)  # Second subplot
ax2 = val_counts.sort_values().plot(
    kind='bar',
    color='lightcoral',
    edgecolor='black',
    width=0.8
)
# Add count labels above bars
for p in ax2.patches:
    ax2.annotate(str(p.get_height()), 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 5), 
                textcoords='offset points')
plt.xlabel("Class Label", fontsize=12)
plt.ylabel("Frequency [count]", fontsize=12)
plt.title("Validation Dataset Class Distribution", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.grid(axis='y', alpha=0.3)  # Add light grid lines

plt.tight_layout()  # Adjust spacing between subplots
plt.show()  # Display the figure

batch_size = 32

# Create training data loader with shuffling for better training
train_loader = DataLoader(training_dataset, batch_size, shuffle=True)

# Create validation data loader without shuffling for consistent evaluation
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

# Print types of the created data loaders
print(type(train_loader))
print(type(val_loader))

data_iter = iter(train_loader)
images, labels = next(data_iter)

# Print image batch shape (batch_size, channels, height, width)
print(f"Image batch shape: {images.shape} (batch_size, channels, height, width)")

# Print label batch shape (batch_size,)
print(f"Label batch shape: {labels.shape} (batch_size,)")

model = torch.nn.Sequential(
    # First convolutional block
    nn.Conv2d(3, 8, kernel_size=3, padding=1),  # Input channels=3 (RGB), output=8
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # Reduces spatial dimensions by half
    
    # Flatten and output layer
    nn.Flatten(),  # Prepares features for dense layer
    nn.Linear(8 * 112 * 112, 4)  # Output layer with 4 classes (diseases + healthy)
)
print(model)


optimizer = optim.Adam(model.parameters(), lr=0.01)

# Print details of our training configuration
print("Training Configuration:")
print(f"Loss Function: {loss_fn}")
print("----------------------")
print(f"Optimizer: {optimizer} (Learning Rate: 0.01)")

height = 224  # Height of input images in pixels
width = 224   # Width of input images in pixels


summary(model, input_size=(batch_size, 3, height, width))

def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=5,
    device='cpu',
    use_train_accuracy=True,
):
    """Train a PyTorch model and validate its performance.
    
    Args:
        model: The neural network model to train
        optimizer: Optimization algorithm (e.g., Adam)
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training iterations (default: 5)
        device: Device to train on ('cpu' or 'cuda')
        use_train_accuracy: Whether to compute training accuracy (default: True)
    
    Returns:
        Tuple of lists containing training/validation losses and accuracies
    """
    # Move model to specified device (CPU/GPU)
    model.to(device)
    
    # Initialize lists to track performance metrics
    train_losses = []  # Training loss per epoch
    val_losses = []    # Validation loss per epoch
    train_accuracies = []  # Training accuracy per epoch
    val_accuracies = []    # Validation accuracy per epoch
    
    # Training loop over specified number of epochs
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Iterate over training batches
        for inputs, labels in train_loader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            # Calculate loss
            loss = loss_fn(outputs, labels)
            # Backward pass (compute gradients)
            loss.backward()
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            
            # Calculate training accuracy if enabled
            if use_train_accuracy:
                _, predicted = torch.max(outputs.data, 1)  # Get predicted class
                total_train += labels.size(0)  # Total samples in batch
                correct_train += (predicted == labels).sum().item()  # Correct predictions
        
        # --- VALIDATION PHASE ---
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        # Disable gradient calculation for validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                # Calculate loss
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                
                # Calculate validation accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # --- METRICS CALCULATION ---
        # Average losses over all batches
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate accuracies (percentage)
        if use_train_accuracy:
            train_acc = 100 * correct_train / total_train
            train_accuracies.append(train_acc)
        val_acc = 100 * correct_val / total_val
        val_accuracies.append(val_acc)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}', end=' ')
        if use_train_accuracy:
            print(f'- Train Acc: {train_acc:.2f}%', end=' ')
        print(f'- Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Execute training with default 5 epochs
train_losses, val_losses, train_accuracies, val_accuracies = train(
    model, optimizer, loss_fn, train_loader, val_loader, epochs=5
)



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss over epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Accuracy over epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

def predict(model, data_loader):
    model.eval()  # Set model to evaluation mode
    all_probs = []  # Store all probabilities
    
    # No gradient calculation needed for prediction
    with torch.no_grad():
        for images, _ in data_loader:  # Iterate through data loader
            outputs = model(images)  # Get model outputs
            probs = torch.softmax(outputs, dim=1)  # Convert to probabilities
            all_probs.append(probs)  # Append batch probabilities
            
    return torch.cat(all_probs, dim=0)  # Concatenate all batch probabilities

# Get validation set probabilities and predictions
probabilities_val = predict(model, val_loader)
predictions_val = torch.argmax(probabilities_val, dim=1)  # Get class with highest probability

print(predictions_val)  # Print predicted classes

targets_val = torch.cat(
    [labels for _, labels in tqdm(val_loader, desc="Get Labels")]
)

cm = confusion_matrix(targets_val.cpu(), predictions_val.cpu())

# Create ConfusionMatrixDisplay object with the matrix and class labels
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# Set figure size for better visualization
plt.figure(figsize=(10, 8))

# Plot confusion matrix with blue color map and vertical x-axis labels
disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")

# Display the plot
plt.show()

report = classification_report(targets_val.cpu(), predictions_val.cpu(), target_names=classes)
print(report)


test_dir = os.path.join('crop pictures', 'test')

# Print the test directory path
print(test_dir)

def file_to_confidence(model, datadir, filename, transform_pipeline):
    # Construct the full file path for the image
    file_path = os.path.join(datadir, filename)
    # Open the image file
    image = PIL.Image.open(file_path)
    # Apply the transformation pipeline to the image
    transformed = transform_pipeline(image)
    # Add a batch dimension to the transformed image
    unsqueezed = transformed.unsqueeze(0)
    # Move the image tensor to the appropriate device (CPU or GPU)
    image_cuda = unsqueezed.to(device)

    # Set the model to evaluation mode
    model.eval()
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Get the raw output from the model
        model_raw = model(image_cuda)
        # Apply softmax to get confidence scores for each class
        confidence = torch.nn.functional.softmax(model_raw, dim=1)

    # Create a DataFrame to store the filename and confidence scores
    conf_df = pd.DataFrame([[filename] + confidence.tolist()[0]])
    # Set the column names to include the image ID and class names
    conf_df.columns = ["ID"] + training_dataset.classes

    # Return the DataFrame containing the confidence scores
    return conf_df

blight_train_dir = os.path.join('crop pictures','train','blight')
blight_images = os.listdir(blight_train_dir)

file_to_confidence(model, blight_train_dir, blight_images[7], transform)

pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Final prediction to hackathon
small_dfs = []

for filename in tqdm(os.listdir(test_dir), desc="Predicting on test set"):
    small_dfs.append(
        file_to_confidence(model, test_dir, filename, transform)
    )

confidence_df = pd.concat(small_dfs)

# Remove file extension from ID column
confidence_df['ID'] = confidence_df['ID'].str.split('.').str[0]

confidence_df = confidence_df.sort_values("ID").reset_index(drop=True)
confidence_df.head()

confidence_df.to_csv("submission.csv", index=False)



