import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set page configuration to use full width
st.set_page_config(page_title="Fish Classification", page_icon="🐟", layout="wide")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",  
        options=["Home", "Upload Image", "Classify", "Model Insights", "About"],
        icons=["house", "cloud-upload", "search", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#E6E6FA"},
            "icon": {"color": "#FF00FF", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "#333",
                "padding": "10px",
                "border-radius": "8px",
            },
            "nav-link-selected": {"background-color": "#DDA0DD", "color": "white"},
        }
    )

# Display selected section
if selected == "Home":
    st.markdown("<h1 style='color: #8A2BE2;'>🏠 Home</h1>", unsafe_allow_html=True)
    st.write("Welcome to the Multiclass Fish Image Classification App!")

elif selected == "Upload Image":
    st.markdown("<h1 style='color: #8A2BE2;'>📤 Upload Image</h1>", unsafe_allow_html=True)
    st.write("Upload a fish image for classification.")

elif selected == "Classify":
    st.markdown("<h1 style='color: #8A2BE2;'>🔍 Classify</h1>", unsafe_allow_html=True)
    st.write("The model will classify the fish image into its respective category.")

elif selected == "Model Insights":
    st.markdown("<h1 style='color: #8A2BE2;'>📊 Model Insights</h1>", unsafe_allow_html=True)
    st.write("View model performance, accuracy, and other insights.")

elif selected == "About":
    st.markdown("<h1 style='color: #8A2BE2;'>ℹ️ About</h1>", unsafe_allow_html=True)
    st.write("Learn more about this project and its objectives.")


if selected == "Home":
    # Create columns for text and video
    col1, col2 = st.columns([2, 1])  # Adjust width (2: text, 1: video)

    with col1:
        # Title
        st.markdown("<h1 style='color: #C71585;'>🎯 Multiclass Fish Image Classification</h1>", unsafe_allow_html=True)

        # Project Overview
        st.markdown("<h2 style='color: #DB7093;'>🔍 Project Overview</h2>", unsafe_allow_html=True)
        st.write("""
        This **Multiclass Fish Image Classification** project focuses on **identifying different fish species** using Deep Learning.
        The model has been trained on multiple fish categories using **MobileNet**, a powerful pre-trained Convolutional Neural Network (CNN).
        """)

        # Features of the Project
        st.markdown("<h2 style='color: #DB7093;'>📌 Features of This Project</h2>", unsafe_allow_html=True)
        st.markdown("""
        - 🐟 **Classifies multiple fish species** using AI-powered deep learning.  
        - 🎯 **Trained with MobileNet architecture** for fast and accurate predictions.  
        - 📷 **Allows users to upload images** and get real-time classification results.  
        - 📊 **Displays confidence scores** to show model certainty.  
        - 🚀 **User-friendly Streamlit interface** with sidebar navigation.  
        """)

        # Technologies & Languages Used
        st.markdown("<h2 style='color: #DB7093;'>🛠 Technologies & Tools Used</h2>", unsafe_allow_html=True)
        st.markdown("""
        - **Programming Language:** Python 🐍  
        - **Framework:** TensorFlow & Keras 🔥  
        - **Web App:** Streamlit 🌐  
        - **Model Architecture:** MobileNet (Pre-trained CNN) 🧠  
        - **Dataset Processing:** NumPy, Pandas  
        - **Visualization:** Matplotlib, Seaborn  
        """)

        # How We Built This Project
        st.markdown("<h2 style='color: #DB7093;'>🔄 How This Project Was Developed</h2>", unsafe_allow_html=True)
        st.markdown("""
        1. 📥 **Collected & Preprocessed Fish Images**  
        2. 🔍 **Used Data Augmentation to Improve Model Generalization**  
        3. 🏋️ **Trained MobileNet Model on Fish Dataset**  
        4. 📊 **Evaluated Model Accuracy, Precision, and Recall**  
        5. 🌐 **Deployed the Model Using Streamlit for Real-Time Classification**  
        """)

    with col2:
        # Space for Video
        st.subheader("📽️ Video")
        video_path = "/Users/arul/Documents/Screen Recording 2025-03-24 at 10.23.15 PM.mov"  # Replace with your video file
        st.video(video_path)

# Upload image

if selected == "Upload Image":
    # Title with Color
    st.markdown("<h1 style='color: #C71585;'>📤 Upload an Image for Classification</h1>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image for model input
        img_array = np.array(image.resize((224, 224))) / 255.0  # Resize and normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Load trained model (Correct File Path)
        model_path = "/Users/arul/Documents/VASUKI/projects/mobilenet_fish_final.keras"
        model = tf.keras.models.load_model(model_path)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Manually define class labels (Update based on your dataset)
        class_labels = [
            "animal fish", "animal fish bass", "fish sea_food black_sea_sprat",
            "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel",
            "fish sea_food red_mullet", "fish sea_food red_sea_bream",
            "fish sea_food sea_bass", "fish sea_food shrimp",
            "fish sea_food striped_red_mullet", "fish sea_food trout"
        ]

        # Get predicted label
        predicted_label = class_labels[predicted_class]

        # Display prediction
        st.subheader(f"🎯 Predicted Fish Species: **{predicted_label}**")
        st.write(f"🔍 Confidence Score: **{np.max(prediction) * 100:.2f}%**")


# Classify

if selected == "Classify":
    # Title
    st.markdown("<h1 style='color: #C71585;'>🔍 Classify Fish Species</h1>", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload a fish image for classification...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        img_array = np.array(image.resize((224, 224))) / 255.0  # Resize & Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Load trained model (Correct File Path)
        model_path = "/Users/arul/Documents/VASUKI/projects/mobilenet_fish_final.keras"
        model = tf.keras.models.load_model(model_path)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Define class labels (Make sure this matches your dataset)
        class_labels = [
            "animal fish", "animal fish bass", "fish sea_food black_sea_sprat",
            "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel",
            "fish sea_food red_mullet", "fish sea_food red_sea_bream",
            "fish sea_food sea_bass", "fish sea_food shrimp",
            "fish sea_food striped_red_mullet", "fish sea_food trout"
        ]

        # Get the predicted label
        predicted_label = class_labels[predicted_class]
        confidence_score = np.max(predictions) * 100  # Convert to percentage

        # Display prediction
        st.subheader(f"🎯 Predicted Fish Species: **{predicted_label}**")
        st.write(f"🔍 Confidence Score: **{confidence_score:.2f}%**")

        # Display confidence scores for all classes
        st.subheader("📊 Confidence Scores for All Classes")
        for i, label in enumerate(class_labels):
            st.write(f"**{label}:** {predictions[0][i] * 100:.2f}%")

# Model Insights

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# Define the dataset path (Update with your actual dataset path)
train_path = "/Users/arul/Documents/VASUKI/projects/Dataset/images.cv_jzk6llhf18tm3k0kyttxz/data/train"
val_path = "/Users/arul/Documents/VASUKI/projects/Dataset/images.cv_jzk6llhf18tm3k0kyttxz/data/val"

# Load datasets using `image_dataset_from_directory`
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'  # Use 'int' for integer labels or 'categorical' for one-hot encoding
)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)


if selected == "Model Insights":
    # Title
    st.markdown("<h1 style='color: #C71585;'>📊 Model Performance & Insights</h1>", unsafe_allow_html=True)

    # Display key performance metrics
    st.subheader("🔹 Model Evaluation Metrics")
    st.markdown("""
    - **Final Validation Accuracy:** 98.7%  
    - **Final Validation Loss:** 0.05  
    - **Best Performing Model:** MobileNet  
    """)

    # Model Comparison Table
    st.subheader("📌 Model Comparison")
    model_results = {
        "Model": ["CNN (Scratch)", "VGG16", "ResNet50", "MobileNet", "InceptionV3", "EfficientNetB0"],
        "Validation Accuracy": [0.864, 0.786, 0.319, 0.987, 0.961, 0.171],
        "Validation Loss": [0.388, 0.933, 1.961, 0.050, 0.130, 2.309]
    }

    df_results = pd.DataFrame(model_results)
    df_results = df_results.sort_values(by="Validation Accuracy", ascending=False)
    st.dataframe(df_results)

    # Accuracy & Loss Bar Chart
    st.subheader("📊 Accuracy & Loss Comparison")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Shortened model names
    short_model_names = ["CNN", "VGG16", "ResNet50", "MobileNet", "InceptionV3", "EffNetB0"]

    # Accuracy Plot
    ax[0].bar(short_model_names, df_results["Validation Accuracy"], color=["blue", "green", "red", "purple", "orange", "cyan"])
    ax[0].set_title("Validation Accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_ylim(0, 1)
    ax[0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

    # Loss Plot
    ax[1].bar(short_model_names, df_results["Validation Loss"], color=["blue", "green", "red", "purple", "orange", "cyan"])
    ax[1].set_title("Validation Loss")
    ax[1].set_ylabel("Loss")
    ax[1].tick_params(axis='x', rotation=90)  # Rotate x-axis labels

    # Display plots
    st.pyplot(fig)


    # Confusion Matrix
    st.subheader("🛠 Confusion Matrix of Best Model")
    st.write("This shows how well the model classified each fish species.")

    # Load trained model
    model_path = "/Users/arul/Documents/VASUKI/projects/mobilenet_fish_final.keras"
    model = tf.keras.models.load_model(model_path)

    # Get true labels from `val_data`
    y_true = []
    y_pred_augmented = []

    # Iterate through validation dataset to extract true labels and predictions
    for images, labels in val_data:
        y_true.extend(np.argmax(labels.numpy(), axis=1))  # Convert one-hot encoding to class indices
        predictions = model.predict(images)  # Get predictions
        y_pred_augmented.extend(np.argmax(predictions, axis=1))  # Convert predictions to class indices

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred_augmented = np.array(y_pred_augmented)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_augmented)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

# About

if selected == "About":
    # Create columns for text and image
    col1, col2 = st.columns([2, 1])  # Text takes 2 parts, Image takes 1 part

    with col1:
        # Title
        st.markdown("<h1 style='color: #C71585;'>ℹ️ About This Project</h1>", unsafe_allow_html=True)

        # Advantages of This Model
        st.markdown("<h2 style='color: #DB7093;'>🚀 Advantages of This Model</h2>", unsafe_allow_html=True)
        st.markdown("""
        - ✅ **Fast & Efficient:** Uses a pre-trained MobileNet model for quick predictions.  
        - 🎯 **High Accuracy:** Achieved **98.7% validation accuracy** with deep learning techniques.  
        - 🌐 **Web-Based Interface:** User-friendly **Streamlit app** for easy access.  
        - 📊 **Comparison of Multiple Models:** Evaluated CNN, VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0.  
        - 🔍 **Handles Multiple Fish Species:** Trained on a dataset with various fish categories.  
        """)

        # Future Enhancements
        st.markdown("<h2 style='color: #DB7093;'>🔮 Future Enhancements</h2>", unsafe_allow_html=True)
        st.markdown("""
        - 🧠 **Improve Model Generalization:** Train on a larger, more diverse dataset.  
        - 📈 **Integrate More Models:** Try Vision Transformers (ViTs) and advanced architectures.  
        - 🎥 **Live Camera Integration:** Allow users to classify fish species using a webcam.  
        - ☁ **Cloud Deployment:** Host the model on **AWS, Google Cloud, or Azure** for scalability.  
        - 📱 **Mobile App Integration:** Develop an Android/iOS app for on-the-go classification.  
        """)

    with col2:
        # Space for an Image
        st.image("/Users/arul/Documents/VASUKI/projects/81f8Sj1ie5L._AC_UF894,1000_QL80_.jpg", caption="Fish Classification Model", use_container_width=True)
    
    # Real-World Applications
    st.markdown("<h2 style='color: #DB7093;'>🌍 Real-World Applications</h2>", unsafe_allow_html=True)
    st.markdown("""
    - 🌊 **Marine Biology:** Helps researchers **identify & classify marine species** efficiently.  
    - 🎣 **Fisheries & Aquaculture:** Assists in **fish species identification** for sustainable fishing.  
    - 🏪 **Retail & Food Industry:** Automates **seafood classification** in supermarkets & restaurants.  
    - 🏛 **Educational Use:** Helps students & researchers learn about **AI-based image classification**.  
    """)

    # Challenges Faced & Solutions
    st.markdown("<h2 style='color: #DB7093;'>⚠️ Challenges Faced & Solutions</h2>", unsafe_allow_html=True)
    st.markdown("""
    - ❌ **Class Imbalance Issue:** Solved by **data augmentation & class weighting**.  
    - 📏 **Different Image Resolutions:** Standardized images to **224x224 pixels** for consistency.  
    - 🏋️ **Overfitting in Small Dataset:** Used **dropout layers & transfer learning** to improve generalization.  
    - 🕒 **Training Time:** Optimized with **pre-trained MobileNet** instead of training from scratch.  
    """)

    # Conclusion
    st.markdown("<h2 style='color: #DB7093;'>🔚 Conclusion</h2>", unsafe_allow_html=True)
    st.markdown("""
    This project successfully implemented a **deep learning model** for fish species classification.  
    With **high accuracy & real-time predictions**, this model is a step towards AI-driven solutions in marine research and commercial applications.  
    Future improvements will make it even more scalable and versatile for different use cases.  
    """)
