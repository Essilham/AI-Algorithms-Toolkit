import streamlit as st

# Title
st.title("AI Model Selection Tool")
st.write("""
This interactive tool helps you select the appropriate AI tool, machine learning model, 
or deep learning model based on your project requirements.
""")

# Questionnaire
st.sidebar.header("Answer the Following Questions")

# 1. Task Type
task_type = st.sidebar.selectbox(
    "What is your task?",
    ["Classification", "Regression", "Clustering", "Other"]
)

# 2. Data Type
data_type = st.sidebar.selectbox(
    "What type of data are you working with?",
    ["Structured (tabular)", "Images", "Text", "Time-Series", "Other"]
)

# 3. Data Size
data_size = st.sidebar.radio(
    "How much data do you have?",
    ["Small (<1,000 samples)", "Medium (1,000-100,000 samples)", "Large (>100,000 samples)"]
)

# 4. Interpretability
interpretability = st.sidebar.radio(
    "Do you need interpretability?",
    ["Yes, it's critical", "No, I just need high performance"]
)

# 5. Computational Resources
resources = st.sidebar.radio(
    "Do you have access to GPUs?",
    ["Yes", "No"]
)

# Recommendations based on answers
st.subheader("Recommendations")

# AI Tool Recommendations
if data_type in ["Images", "Text"] and data_size == "Small (<1,000 samples)":
    st.write("**AI Tools Recommendation**: Consider using prebuilt AI tools or APIs like:")
    st.write("- Google AutoML")
    st.write("- Hugging Face Transformers")
    st.write("- OpenAI GPT models")
elif data_type == "Structured (tabular)" and interpretability == "Yes, it's critical":
    st.write("**AI Tools Recommendation**: Consider tools like:")
    st.write("- DataRobot for automated model building with explainability.")
    st.write("- SHAP (SHapley Additive exPlanations) for model interpretability.")

# Machine Learning Model Recommendations
if task_type == "Classification" and data_type == "Structured (tabular)":
    if interpretability == "Yes, it's critical":
        st.write("**Machine Learning Model Recommendation**: Decision Tree or Logistic Regression.")
    else:
        st.write("**Machine Learning Model Recommendation**: Random Forest or Gradient Boosting.")
elif task_type == "Regression":
    if data_size == "Small (<1,000 samples)":
        st.write("**Machine Learning Model Recommendation**: Linear Regression or Ridge Regression.")
    else:
        st.write("**Machine Learning Model Recommendation**: Gradient Boosting (e.g., XGBoost, LightGBM).")
elif task_type == "Clustering":
    st.write("**Machine Learning Model Recommendation**: K-Means or DBSCAN.")

# Deep Learning Recommendations
if data_type in ["Images", "Text"] and data_size == "Large (>100,000 samples)":
    if data_type == "Images":
        st.write("**Deep Learning Model Recommendation**: Convolutional Neural Networks (CNNs).")
    elif data_type == "Text":
        st.write("**Deep Learning Model Recommendation**: Transformers (e.g., BERT, GPT).")
    if resources == "Yes":
        st.write("**Tip**: Train on GPUs for faster performance.")
    else:
        st.write("**Tip**: Use pretrained models to save time and computational costs.")
elif data_type == "Time-Series":
    st.write("**Deep Learning Model Recommendation**: Recurrent Neural Networks (RNNs) or LSTMs.")

# Final Summary
st.write("""
### Summary:
- This tool provides model recommendations tailored to your project needs.
- Make adjustments to the questionnaire to refine your choices.
""")
