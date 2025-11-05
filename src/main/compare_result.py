import pandas as pd
import matplotlib.pyplot as plt

# Đọc file log
df = pd.read_csv("out/training_log.csv")

epochs = df["epoch"]

# ===== 1️⃣ Accuracy =====
plt.figure()
plt.plot(epochs, df["accuracy"], label="Training Accuracy")
plt.plot(epochs, df["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ===== 2️⃣ Loss =====
plt.figure()
plt.plot(epochs, df["loss"], label="Training Loss")
plt.plot(epochs, df["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

