import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 11))

# Accuracy values
train_acc = [50, 55, 60, 65, 70, 73, 75, 76, 77, 78]
val_acc   = [48, 53, 58, 63, 68, 71, 73, 74, 75, 76]

# Loss values
train_loss = [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35]
val_loss   = [1.3, 1.1, 1.0, 0.9, 0.85, 0.75, 0.7, 0.68, 0.65, 0.63]

# ---------------------------
# Graph 1: Accuracy
# ---------------------------
plt.figure()
plt.plot(epochs, train_acc, marker='o', label='Training Accuracy')
plt.plot(epochs, val_acc, marker='o', label='Validation Accuracy')

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.savefig("accuracy_comparison.png")

# ---------------------------
# Graph 2: Loss
# ---------------------------
plt.figure()
plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='o', label='Validation Loss')

plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_comparison.png")

plt.show()