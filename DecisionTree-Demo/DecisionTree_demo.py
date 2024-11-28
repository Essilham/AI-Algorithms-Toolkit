import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modules.tree_visualizer import visualize_tree
from modules.data_loader import load_sample_data, load_synthetic_data

# App Title
st.title("Interactive Decision Tree Visualization")
st.write("Explore Decision Tree classification with interactive controls!")

# Sidebar Configuration
criterion = st.sidebar.selectbox("Select Splitting Criterion", ["gini", "entropy"])
max_depth = st.sidebar.slider("Max Depth", 1, 20, value=5)
split_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.9, value=0.7)
random_state = st.sidebar.slider("Random State", 0, 100, value=42)

# Dataset Selection
st.sidebar.write("## Dataset Options")
dataset_option = st.sidebar.selectbox("Choose Dataset", ["Iris Dataset", "Synthetic Data"])

# Load Data
if dataset_option == "Iris Dataset":
    data = load_sample_data()
else:
    data = load_synthetic_data()

# Display Dataset
st.write("### Dataset Preview")
st.dataframe(data.head())

# Split Data
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state)

# Train Decision Tree
model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Visualize Decision Tree
st.write("### Decision Tree Visualization")
visualize_tree(model, X.columns)

