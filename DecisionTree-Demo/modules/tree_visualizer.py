import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import streamlit as st

def visualize_tree(model, feature_names):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, feature_names=feature_names, class_names=True, filled=True, ax=ax)
    st.pyplot(fig)
