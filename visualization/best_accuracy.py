import matplotlib.pyplot as plt

# Updated test accuracies for the models
models = ["FNN", "CNN", "Transformer", "BERT (Pre-trained)"]
accuracies = [0.7208, 0.7487, 0.7589, 0.7995]

# Sort the models and accuracies for visualization
sorted_indices = sorted(range(len(accuracies)), key=lambda k: accuracies[k])
models_sorted = [models[i] for i in sorted_indices]
accuracies_sorted = [accuracies[i] for i in sorted_indices]

# Create the updated plot
plt.figure(figsize=(8, 5))
plt.barh(models_sorted, accuracies_sorted, color="skyblue", edgecolor="black")
plt.xlabel("Best Test Accuracy", fontsize=12)
plt.title("Comparison of Best Test Accuracy by Model", fontsize=14)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.xlim(0.7, 0.85)  # Adjust limits based on updated accuracy range

# Annotate accuracy values
for i, acc in enumerate(accuracies_sorted):
    plt.text(acc + 0.002, i, f"{acc:.4f}", va='center', fontsize=10)

# Display the updated plot
plt.tight_layout()
plt.show()
