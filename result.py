import matplotlib.pyplot as plt
import pandas as pd

result = {
    "Logic Regression": {"Recall": 0.729, "Precision": 0.624, "F1": 0.673},
    "Decision Tree": {"Recall": 0.694, "Precision": 0.624, "F1": 0.6962},
    "SVM": {"Recall": 0.704, "Precision": 0.705, "F1": 0.705},
    "KNN": {"Recall": 0.647, "Precision": 0.643, "F1": 0.645},
    "MLP": {"Recall": 0.770, "Precision": 0.597, "F1": 0.672},
}

df = pd.DataFrame(result).T

plt.figure(figsize=(9, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bar_plot = df.plot(kind='bar', figsize=(9, 6), width=0.7, color=colors, edgecolor='black')

plt.title("Comparison of Model Performance (Recall, Precision, F1)", fontsize=14, pad=15)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Model", fontsize=12)

plt.ylim(0.5, 0.8)
plt.xticks(rotation=30, ha='right')

for container in bar_plot.containers:
    bar_plot.bar_label(container, fmt='%.3f', label_type='edge', fontsize=9, padding=2)

plt.legend(title="Metric", fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("RQ1-result.png", dpi=300)
plt.show()