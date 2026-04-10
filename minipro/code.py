import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from matplotlib.colors import ListedColormap

# ✅ Create outputs folder
os.makedirs("outputs", exist_ok=True)

# =========================================================
# 🔹 Load Dataset
# =========================================================
wine = load_wine()
dataset = pd.DataFrame(columns=wine.feature_names, data=wine.data)
dataset['target'] = wine.target

# Features and Target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# =========================================================
# 🔹 Preprocessing
# =========================================================
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# =========================================================
# 🔹 1. 3D Visualization (Original Data)
# =========================================================
fig1 = plt.figure(figsize=(7,5))
ax = fig1.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='rainbow', alpha=0.7)

ax.set_xlabel(wine.feature_names[0])
ax.set_ylabel(wine.feature_names[1])
ax.set_zlabel(wine.feature_names[2])
ax.set_title('Original Wine Dataset (3D)')

fig1.savefig("outputs/output1_original_3d.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig1)

# =========================================================
# 🔹 2. Random Forest WITHOUT LDA
# =========================================================
X_train_2D = X_train[:, :2]
X_test_2D = X_test[:, :2]

rf_without_lda = RandomForestClassifier(max_depth=2, random_state=0)
rf_without_lda.fit(X_train_2D, y_train)

# Predictions
y_pred1 = rf_without_lda.predict(X_test_2D)

# Accuracy
acc1 = accuracy_score(y_test, y_pred1)
print("Accuracy without LDA:", acc1)

# Performance Metrics
cm1 = confusion_matrix(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1, average='weighted')
recall1 = recall_score(y_test, y_pred1, average='weighted')
f1_1 = f1_score(y_test, y_pred1, average='weighted')

print("\n=== WITHOUT LDA ===")
print("Confusion Matrix:\n", cm1)
print("Precision:", precision1)
print("Recall:", recall1)
print("F1 Score:", f1_1)

# Decision Boundary
x_min, x_max = X_train_2D[:,0].min() - 1, X_train_2D[:,0].max() + 1
y_min, y_max = X_train_2D[:,1].min() - 1, X_train_2D[:,1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = rf_without_lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

fig2 = plt.figure(figsize=(7,5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
plt.scatter(X_train_2D[:,0], X_train_2D[:,1], c=y_train, cmap='rainbow')

plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])
plt.title('Random Forest Decision Boundary Without LDA')

plt.tight_layout()
fig2.savefig("outputs/output2_without_lda.png", dpi=300)
plt.show()
plt.close(fig2)

# =========================================================
# 🔹 3. Apply LDA
# =========================================================
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

rf_with_lda = RandomForestClassifier(max_depth=2, random_state=0)
rf_with_lda.fit(X_train_lda, y_train)

# Predictions
y_pred2 = rf_with_lda.predict(X_test_lda)

# Accuracy
acc2 = accuracy_score(y_test, y_pred2)
print("\nAccuracy with LDA:", acc2)

# Performance Metrics
cm2 = confusion_matrix(y_test, y_pred2)
precision2 = precision_score(y_test, y_pred2, average='weighted')
recall2 = recall_score(y_test, y_pred2, average='weighted')
f1_2 = f1_score(y_test, y_pred2, average='weighted')

print("\n=== WITH LDA ===")
print("Confusion Matrix:\n", cm2)
print("Precision:", precision2)
print("Recall:", recall2)
print("F1 Score:", f1_2)

# Decision Boundary
x_min, x_max = X_train_lda[:,0].min() - 1, X_train_lda[:,0].max() + 1
y_min, y_max = X_train_lda[:,1].min() - 1, X_train_lda[:,1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = rf_with_lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig3 = plt.figure(figsize=(7,5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
plt.scatter(X_train_lda[:,0], X_train_lda[:,1], c=y_train, cmap='rainbow')

plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('Random Forest Decision Boundary With LDA')

plt.tight_layout()
fig3.savefig("outputs/output3_with_lda.png", dpi=300)
plt.show()
plt.close(fig3)

# =========================================================
# 🔹 Save Results
# =========================================================
with open("outputs/performance.txt", "w") as f:
    f.write("=== WITHOUT LDA ===\n")
    f.write(f"Accuracy: {acc1:.4f}\n")
    f.write(f"Precision: {precision1:.4f}\n")
    f.write(f"Recall: {recall1:.4f}\n")
    f.write(f"F1 Score: {f1_1:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm1}\n\n")

    f.write("=== WITH LDA ===\n")
    f.write(f"Accuracy: {acc2:.4f}\n")
    f.write(f"Precision: {precision2:.4f}\n")
    f.write(f"Recall: {recall2:.4f}\n")
    f.write(f"F1 Score: {f1_2:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm2}\n")

