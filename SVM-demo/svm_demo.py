import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import streamlit as st

# Streamlit App Title
st.title("Interactive SVM Visualization")
st.write("Explore how Support Vector Machines (SVMs) classify data with different kernels and parameters!")

# Sidebar Controls
kernel = st.sidebar.selectbox("Select Kernel", ["linear", "rbf", "poly"])
regularization = st.sidebar.slider("Regularization Parameter (C)", 0.1, 10.0, step=0.1, value=1.0)
dataset_type = st.sidebar.selectbox("Select Dataset", ["Linearly Separable", "Non-Linearly Separable (Moons)", "Non-Linearly Separable (Circles)"])
noise = st.sidebar.slider("Dataset Noise", 0.0, 1.0, step=0.1, value=0.1)
random_state = st.sidebar.slider("Random State", 0, 100, step=1, value=42)

# Generate the dataset
if dataset_type == "Linearly Separable":
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2, n_redundant=0, 
        n_clusters_per_class=1, class_sep=2, random_state=random_state
    )
elif dataset_type == "Non-Linearly Separable (Moons)":
    X, y = make_moons(n_samples=200, noise=noise, random_state=random_state)
else:
    X, y = make_circles(n_samples=200, noise=noise, factor=0.5, random_state=random_state)

# Train the SVM model
svm = SVC(kernel=kernel, C=regularization)
svm.fit(X, y)

# Create a mesh grid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict on the mesh grid
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
fig, ax = plt.subplots(figsize=(8, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['red', 'blue']

ax.contourf(xx, yy, Z, alpha=0.6, cmap=cmap_light)

# Plot the training points
for idx, color in enumerate(cmap_bold):
    ax.scatter(X[y == idx, 0], X[y == idx, 1], c=color, label=f"Class {idx}", edgecolor='k')

# Highlight support vectors
ax.scatter(
    svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
    s=100, facecolors='none', edgecolors='black', label="Support Vectors"
)

ax.set_title(f"SVM Decision Boundary (Kernel: {kernel}, C: {regularization})")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

# Display the plot
st.pyplot(fig)

