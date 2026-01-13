import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
import time

# --- Configuration ---
DATASET_DIRS = ['./dataset_1', './dataset_2']  # The folders you created
NUM_CLASSES = 102
BATCH_SIZE = 32
EPOCHS = 20  # Adjustable based on convergence
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """
    Detailed Preprocessing:
    1. Resize: Scales image so smaller edge is 256.
    2. Crop: Extracts 224x224 patch (Random for train, Center for val/test).
    3. ToTensor: Converts [0, 255] range to [0.0, 1.0].
    4. Normalize: Standardizes using ImageNet mean/std.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),  # Augmentation
            transforms.RandomHorizontalFlip(),  # Augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }


def initialize_vgg19(num_classes):
    # Load VGG19 with pretrained weights
    print("Loading VGG19...")
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    # Freeze feature extraction layers to use Transfer Learning
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final classification layer
    # VGG classifier block is a Sequential module; index 6 is the final Linear layer
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model.to(DEVICE)


def evaluate_test(model, test_loader, criterion, dataset_size):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # No gradient needed for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / dataset_size
    test_acc = running_corrects.double() / dataset_size

    print(f"Final Test Result -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    # Check assignment requirement (>70%)
    if test_acc > 0.70:
        print("SUCCESS: Model accuracy is greater than 70%!")
    else:
        print("WARNING: Model accuracy is below 70%.")


def train_model(data_dir, run_name):
    transforms_map = get_transforms()

    # Load Data
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transforms_map[x])
                      for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=(x == 'train'), num_workers=4)
                   for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    # Setup Model, Loss, Optimizer
    model = initialize_vgg19(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    # Optimize only the classifier parameters since features are frozen
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

    # Metrics Storage
    history = {'train_loss': [], 'val_loss': [], 'test_loss': [],
               'train_acc': [], 'val_acc': [], 'test_acc': []}

    print(f"\nStarting training for {run_name} on {DEVICE}...")

    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        # Note: Assignment asks for Test graphs too, so we evaluate Test every epoch
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + Optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    plot_results(history, run_name)

    print(f"\nEvaluating {run_name} on TEST set...")
    evaluate_test(model, dataloaders['test'], criterion, dataset_sizes['test'])

    return model


def plot_results(history, run_name):
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Train')
    plt.plot(epochs_range, history['val_acc'], label='Val')
    plt.plot(epochs_range, history['test_acc'], label='Test')
    plt.title(f'{run_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Train')
    plt.plot(epochs_range, history['val_loss'], label='Val')
    plt.plot(epochs_range, history['test_loss'], label='Test')
    plt.title(f'{run_name} Cross-Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{run_name}_metrics.png')
    print(f"Graphs saved as {run_name}_metrics.png")
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    for i, data_dir in enumerate(DATASET_DIRS):
        run_name = f"VGG19_Run_{i + 1}"
        if os.path.exists(data_dir):
            train_model(data_dir, run_name)
        else:
            print(f"Directory {data_dir} not found. Skipping.")
