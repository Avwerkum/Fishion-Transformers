# %%
# Import required libraries
from torchvision.datasets import ImageFolder
from torchvision import transforms
import timm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import os
import numpy as np
import matplotlib.pyplot as plt
import config
import time

# %%
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if CUDA is not available or if cuda:0 is not available, find another available CUDA device or use CPU
if str(device) == "cpu":
    print("No CUDA devices available. Using CPU.")
else:
    try:
        # try to use cuda:0
        torch.cuda.set_device(device)
        print("Using CUDA device:", device)
    except RuntimeError:
        # if cuda:0 is not available, find another available cuda device or use CPU
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        print(devices)
        for dev in devices:
            try:
                torch.cuda.set_device(dev)
                print(f"Using CUDA device: {dev}")
                device = torch.device(dev)
                break
            except RuntimeError:
                pass
        if str(device) == "cpu":
            print("No CUDA devices available. Using CPU.")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])


trainset = ImageFolder(config.TRAIN_DIR, transform=transform)
testset = ImageFolder(config.TEST_DIR, transform=transform)
valset = ImageFolder(config.VAL_DIR, transform=transform)

# DataLoader
train_loader = DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(valset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
test_loader = DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)


# %%
torch.cuda.reset_peak_memory_stats
start_time = time.time()
     
#import the model and set checkpoint dir
model_name = "efficientnet_b2"
model = getattr(config, model_name)
save_dir = f"{config.SAVE_DIR}/{model_name}"

# Check if save directory exists
if os.path.exists(save_dir):
    pass
else:
    # Save directory does not exist, create it
    os.makedirs(save_dir)


# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

if hasattr(model, 'classifier'):
    # Replace final layer
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, config.NUM_CLASSES)

    # Set the requires_grad attribute of the new fully connected layer to True
    model.classifier.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
else:
    # Replace final layer
    num_features = model.head.in_features
    model.head = torch.nn.Linear(num_features, config.NUM_CLASSES)

    # Set the requires_grad attribute of the new fully connected layer to True
    model.head.requires_grad = True

    optimizer = optim.AdamW([{"params": model.head.parameters(), "lr": config.LEARNING_RATE}], weight_decay=config.WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= config.T_MAX, eta_min=config.ETA_MIN)

# %%
model.to(device)

# Define your early stopping criteria
patience = config.PATIENCE
delta = config.DELTA
best_loss = float('inf')
counter = 0

#Set epochs
start_epoch = 0
num_epochs = config.NUM_EPOCHS
train_accs = []
val_accs = []

# %%

#Training and validating

print(f"Beginning training of {model_name}")

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0
    
    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        
    # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
    
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Calculate the average training loss and accuracy
    train_loss /= len(train_loader)
    train_acc = train_correct / total
    train_accs.append(train_acc)


    # Set the model to evaluation mode
    model.eval()

    y_true = []
    y_pred = []

    val_loss = 0
    val_correct = 0
    total = 0

    # Disable gradient computation during validation
    with torch.no_grad():
        # Loop over the batches in the validation loader
        for inputs, labels in val_loader:
            # Move the inputs and labels to the GPU if available
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Update the validation loss
            val_loss += loss.item() * inputs.size(0)

            # Calculate the number of correct predictions
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Append true and predicted labels to y_true and y_pred lists
            y_true += labels.cpu().numpy().tolist()
            y_pred += predicted.cpu().numpy().tolist()

    # Calculate the average validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = val_correct / total
    val_accs.append(val_acc)
    val_f1score = f1_score(y_true, y_pred, average='weighted')

    # Print the training and validation loss and accuracy for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1score:.4f}")

    # Check if early stopping criteria is met
    if val_loss < best_loss - delta:
        best_loss = val_loss
        best_epoch = epoch + 1 
        counter = 0

        # Save best model
        checkpoint_name = f"best_model{model_name}.pth"
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)

    else:
        counter += 1
        if counter >= patience:
            print(f"Early Stopping after epoch {epoch + 1}, Best epoch = {best_epoch}/{epoch + 1}")

            break

# Report memory usage and training time
max_memory_bytes = torch.cuda.max_memory_allocated()
print("Maximum GPU memory usage: {:.2f} MB".format(max_memory_bytes / 1024 / 1024))
elapsed_time = (time.time() - start_time) / 60
print("Training time: {:.2f} minutes".format(elapsed_time))

# Testing the model

#Load best state from training
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)


model.eval()
test_loss = 0
test_correct = 0
total = 0
y_true = []
y_pred = []

# Disable gradient computation during testing
with torch.no_grad():
    # Loop over the batches in the test loader
    for inputs, labels in test_loader:
        # Move the inputs and labels to the GPU if available
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Update the test loss
        test_loss += loss.item() * inputs.size(0)

        # Calculate the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Append true and predicted labels to y_true and y_pred lists
        y_true += labels.cpu().numpy().tolist()
        y_pred += predicted.cpu().numpy().tolist()

# Calculate the average test loss and accuracy
test_loss /= len(test_loader)
test_acc = test_correct / total

# Calculate the F1 score
f1 = f1_score(y_true, y_pred, average='weighted')

# Print the test loss, accuracy, and F1 score
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, F1 Score: {f1:.4f}")