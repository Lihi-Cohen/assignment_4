from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

results_csv = Path("yolov5/runs/flowers_cls/exp_dataset1/results.csv")
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()
print(df.columns.tolist())

epochs = df["epoch"] + 1

train_loss = df["train/loss"]
val_loss   = df["val/loss"]
test_loss  = df["test/loss"]

train_acc  = df["metrics/train_accuracy"]
val_acc    = df["metrics/val_accuracy_top1"]
test_acc   = df["metrics/test_accuracy_top1"]

# Loss curves
plt.figure(figsize=(7, 4))
plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss,   label="Val loss")
plt.plot(epochs, test_loss,  label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("YOLOv5 loss vs epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("yolo_loss_exp_dataset1.png", dpi=200)   # <-- save
plt.show()

# Accuracy curves
plt.figure(figsize=(7, 4))
plt.plot(epochs, train_acc, label="Train accuracy")
plt.plot(epochs, val_acc,   label="Val accuracy")
plt.plot(epochs, test_acc,  label="Test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("YOLOv5 accuracy vs epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("yolo_accuracy_exp_dataset1.png", dpi=200)  # <-- save
plt.show()
