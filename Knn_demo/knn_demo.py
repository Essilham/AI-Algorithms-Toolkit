import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import streamlit as st

# Streamlit App Title
st.title("Interactive k-NN Visualization")
st.write("Explore the effects of k-NN using L1 (Manhattan) and L2 (Euclidean) distances!")

# Sidebar Controls
distance_metric = st.sidebar.selectbox("Select Distance Metric", ["L1 (Manhattan)", "L2 (Euclidean)"])
k_value = st.sidebar.slider("Select k (Number of Neighbors)", min_value=1, max_value=10, value=1)

# Generate a synthetic dataset
n_samples = st.sidebar.slider("Number of Samples", min_value=50, max_value=500, value=100)
X, y = make_classification(
    n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, n_classes=4, random_state=42
)

# Configure Distance Metric for k-NN
metric = 'manhattan' if distance_metric == "L1 (Manhattan)" else 'euclidean'

# Fit k-NN model
knn = KNeighborsClassifier(n_neighbors=k_value, metric=metric)
knn.fit(X, y)

# Create a mesh grid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict the class for each point in the grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
fig, ax = plt.subplots(figsize=(8, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFD700'])
cmap_bold = ['r', 'g', 'b', 'gold']

ax.contourf(xx, yy, Z, alpha=0.6, cmap=cmap_light)

# Plot training data
for idx, color in enumerate(cmap_bold):
    ax.scatter(X[y == idx, 0], X[y == idx, 1], c=color, label=f'Class {idx}', edgecolor='k')

ax.set_title(f"k-NN Decision Boundary (k={k_value}, Metric={distance_metric})")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

# Display plot
st.pyplot(fig)
