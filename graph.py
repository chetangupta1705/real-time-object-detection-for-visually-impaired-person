import matplotlib.pyplot as plt

# Models used for comparison
models = ['Faster R-CNN', 'YOLOv5', 'SSD MobileNet', 'Proposed']

# Your values (you can slightly adjust if needed)
accuracy = [73.2, 69.8, 72.5, 75.6]

# Create bar graph
plt.figure()
plt.bar(models, accuracy)

plt.title("Performance Evaluation (Accuracy Comparison)")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=20)
plt.tight_layout()

# Save image
plt.savefig("performance_graph.png")

# Show graph
plt.show()