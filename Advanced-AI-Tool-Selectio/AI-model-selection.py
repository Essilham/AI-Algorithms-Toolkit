import streamlit as st


st.set_page_config(
    page_title="AI Model Selection Tool",
    page_icon=":rocket:",
    layout="centered",
    initial_sidebar_state="expanded",
    meta_tags={
        "description": "An interactive tool to select the best AI models, loss functions, and optimizers tailored to your task.",
        "og:title": "AI Model Selection Tool",
        "og:type": "website",
        "og:url": "https://ai-algorithms-toolkit-advances-ai-models.streamlit.app",
        "og:image": "https://path_to_your_thumbnail_image.png",
        "twitter:card": "summary_large_image",
        "twitter:title": "AI Model Selection Tool",
        "twitter:description": "An interactive Streamlit app to explore AI models, loss functions, and optimizers.",
        "twitter:image": "https://path_to_your_thumbnail_image.png",
    }
)


# Title
st.title("Advanced AI Model Selection Tool")
st.write("""
This tool guides you in selecting the best AI tools, models, loss functions, optimizers, 
hyperparameter tuning methods, and metrics based on your project requirements.
""")

# Sidebar: Task Selection
st.sidebar.header("1. Task Type")
task_type = st.sidebar.selectbox(
    "What is your task?",
    ["Classification", "Regression", "Clustering", "Text Generation", "Image Segmentation", "Other"]
)

# Sidebar: Data Type
st.sidebar.header("2. Data Type")
data_type = st.sidebar.selectbox(
    "What type of data are you working with?",
    ["Structured (tabular)", "Images", "Text", "Time-Series", "Other"]
)

# Sidebar: Data Characteristics
st.sidebar.header("3. Data Characteristics")
data_size = st.sidebar.radio(
    "How much data do you have?",
    ["Small (<1,000 samples)", "Medium (1,000-100,000 samples)", "Large (>100,000 samples)"]
)
label_availability = st.sidebar.radio(
    "Do you have labeled data?",
    ["Yes (Supervised)", "No (Unsupervised)", "Partially (Semi-Supervised)"]
)

# Sidebar: Model Preferences
st.sidebar.header("4. Model Preferences")
interpretability = st.sidebar.radio(
    "Do you need interpretability?",
    ["Yes", "No"]
)
computation = st.sidebar.radio(
    "Do you have access to GPUs?",
    ["Yes", "No"]
)

# Recommendations Section
st.subheader("Recommendations")

# Task-Based Recommendations
if task_type == "Classification":
    st.write("### Task: Classification")
    if data_type == "Structured (tabular)":
        st.write("**Model Recommendation**: Random Forest, XGBoost, or LightGBM for structured data.")
        st.write("**Loss Function**: Cross-Entropy Loss.")
    elif data_type == "Images":
        st.write("**Model Recommendation**: Convolutional Neural Networks (CNNs).")
        st.write("**Loss Function**: Categorical Cross-Entropy or Binary Cross-Entropy.")
    elif data_type == "Text":
        st.write("**Model Recommendation**: Transformers (e.g., BERT, GPT).")
        st.write("**Loss Function**: Cross-Entropy Loss for token classification.")
elif task_type == "Regression":
    st.write("### Task: Regression")
    st.write("**Model Recommendation**: Linear Regression, Ridge Regression, or Gradient Boosting.")
    st.write("**Loss Function**: Mean Squared Error (MSE) or Mean Absolute Error (MAE).")
elif task_type == "Image Segmentation":
    st.write("### Task: Image Segmentation")
    st.write("**Model Recommendation**: U-Net or Mask R-CNN.")
    st.write("**Loss Function**: Dice Loss or IoU Loss.")
elif task_type == "Clustering":
    st.write("### Task: Clustering")
    st.write("**Model Recommendation**: K-Means or DBSCAN.")
    st.write("No loss function is required for unsupervised tasks.")

# Optimizer Recommendation
st.subheader("Optimizer Recommendation")
if task_type in ["Classification", "Regression", "Image Segmentation"]:
    if computation == "Yes":
        st.write("**Optimizer Recommendation**: Adam (fast convergence with GPU).")
    else:
        st.write("**Optimizer Recommendation**: SGD with momentum for CPU-friendly optimization.")

# Hyperparameter Tuning
st.subheader("Hyperparameter Tuning")
st.write("**Tools Recommendation**: Optuna, Ray Tune, or Grid Search.")
st.write("**Parameters to Tune**: Learning rate, batch size, number of layers, dropout rate, and weight decay.")

# Metrics Recommendation
st.subheader("Metrics Recommendation")
if task_type == "Classification":
    st.write("**Metrics**: Accuracy, Precision/Recall, F1-Score, ROC-AUC.")
elif task_type == "Regression":
    st.write("**Metrics**: RMSE, MAE, or R².")
elif task_type == "Image Segmentation":
    st.write("**Metrics**: IoU (Intersection over Union), Dice Coefficient.")

# Additional Tools for Explainability
st.subheader("Explainability Tools")
if interpretability == "Yes":
    st.write("**Tools Recommendation**:")
    st.write("- SHAP (SHapley Additive exPlanations)")
    st.write("- LIME (Local Interpretable Model-agnostic Explanations)")
    st.write("- Captum (for deep learning models)")
else:
    st.write("Explainability is not required based on your selection.")

# Advanced Recommendations for Students
st.subheader("Advanced AI Tools and Frameworks")
st.write("""
- **Experiment Tracking**: Use Weights & Biases or Neptune.ai for tracking hyperparameters and performance.
- **Pretrained Models**: Use Hugging Face Transformers for text or torchvision models for images.
- **Model Deployment**: Use TensorFlow Serving, ONNX, or FastAPI for production deployment.
""")
