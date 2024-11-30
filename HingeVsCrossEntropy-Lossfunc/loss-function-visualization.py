import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loss Functions
def hinge_loss(scores, true_label):
    true_score = scores[true_label]
    return sum(max(0, score - true_score + 1) for i, score in enumerate(scores) if i != true_label)

def cross_entropy_loss(scores, true_label):
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores)
    return -np.log(probabilities[true_label]), probabilities

# Simulate Scores
def simulate_scores(num_classes=3):
    return np.random.rand(num_classes) * 5

# App Layout
st.title("Advanced Loss Function Visualization")
st.write("Explore how hinge loss and cross-entropy loss behave in classification tasks.")

# Sidebar
true_label = st.sidebar.selectbox("True Class Label", [0, 1, 2])
predicted_scores = [
    st.sidebar.slider(f"Score for Class {i}", -5.0, 5.0, 0.0, 0.1) for i in range(3)
]

# Display Scores
st.write(f"**True Label:** {true_label}")
st.write(f"**Predicted Scores:** {predicted_scores}")

# Loss Function Selection
loss_function = st.sidebar.selectbox("Select Loss Function", ["Hinge Loss", "Cross-Entropy Loss"])

if loss_function == "Hinge Loss":
    hinge = hinge_loss(predicted_scores, true_label)
    st.subheader("Hinge Loss")
    st.latex(r"L_{\text{hinge}} = \sum_{j \neq y} \max(0, s_j - s_y + 1)")
    st.write(f"**Hinge Loss Value:** {hinge:.2f}")
elif loss_function == "Cross-Entropy Loss":
    cross_entropy, probabilities = cross_entropy_loss(predicted_scores, true_label)
    st.subheader("Cross-Entropy Loss")
    st.latex(r"L_{\text{cross-entropy}} = -\log(p_y)")
    st.write(f"**Cross-Entropy Loss Value:** {cross_entropy:.2f}")
    st.write(f"**Softmax Probabilities:** {probabilities}")

# Loss Landscape Visualization
st.sidebar.subheader("Loss Landscape Visualization")
view_3d = st.sidebar.checkbox("View 3D Loss Landscape")

if view_3d:
    st.subheader("Loss Landscape")
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([hinge_loss([xi, yi, 0], true_label) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("Score for Class 0")
    ax.set_ylabel("Score for Class 1")
    ax.set_zlabel("Hinge Loss")
    st.pyplot(fig)
