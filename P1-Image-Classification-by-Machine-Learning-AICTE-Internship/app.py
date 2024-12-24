import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.title("MobileNetV2 Image Classification")
    st.write("Upload one or more images, and the app will classify them using the MobileNetV2 model (trained on ImageNet).")

    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png"], accept_multiple_files=True)

    if uploaded_files:
        # Load the MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        results = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')  # Ensure the image is RGB
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            # Preprocess the image
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            # Make predictions
            predictions = model.predict(img_array)
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

            # Display predictions
            st.write(f"### Predictions for {uploaded_file.name}:")
            for imagenet_id, label, score in decoded_predictions:
                st.write(f"- **{label.capitalize()}**: {score * 100:.2f}%")

            # Collect predictions for download
            results.append({
                "Image": uploaded_file.name,
                "Top Prediction": f"{decoded_predictions[0][1]} ({decoded_predictions[0][2] * 100:.2f}%)",
                "Second Prediction": f"{decoded_predictions[1][1]} ({decoded_predictions[1][2] * 100:.2f}%)",
                "Third Prediction": f"{decoded_predictions[2][1]} ({decoded_predictions[2][2] * 100:.2f}%)"
            })

            # Graphical display of confidence scores
            st.write("#### Confidence Scores (Top 3)")
            labels = [label.capitalize() for _, label, _ in decoded_predictions]
            scores = [score * 100 for _, _, score in decoded_predictions]
            fig, ax = plt.subplots()
            ax.bar(labels, scores, color=['blue', 'orange', 'green'])
            ax.set_title("Confidence Scores")
            ax.set_ylabel("Confidence (%)")
            st.pyplot(fig)


        if results:
            import pandas as pd
            st.download_button(
                label="Download Predictions as CSV",
                data=pd.DataFrame(results).to_csv(index=False),
                file_name="mobilenet_predictions.csv",
                mime="text/csv",
            )
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    st.write("Upload an image, and the app will classify it using the CIFAR-10 model.")

    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png"], accept_multiple_files=True)

    if uploaded_files:
        model = tf.keras.models.load_model('cifar10_model.h5')
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        results = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB format
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            
            img = image.resize((32, 32))
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            st.write(f"### Predictions for {uploaded_file.name}:")
            st.write(f"- **Predicted Class:** {class_names[predicted_class]}")
            st.write(f"- **Confidence:** {confidence * 100:.2f}%")
            
            # Graphical display of confidence scores
            fig, ax = plt.subplots()
            ax.bar(class_names, predictions[0])
            ax.set_title("Confidence Scores")
            ax.set_xlabel("Classes")
            ax.set_ylabel("Confidence")
            st.pyplot(fig)
            
            results.append({"Image": uploaded_file.name, "Predicted Class": class_names[predicted_class], "Confidence": confidence * 100})

        # Provide download option for batch results
        if results:
            st.download_button(
                label="Download Predictions as CSV",
                data=pd.DataFrame(results).to_csv(index=False),
                file_name="cifar10_predictions.csv",
                mime="text/csv",
            )

            

# Welcome page
def welcome_page():
    st.title("Welcome to Image Classification App")
    st.write(
        """
        This app allows you to classify images using two models:
        - **MobileNetV2**: Pre-trained on ImageNet for generic object classification.
        - **CIFAR-10 Model**: Trained specifically on CIFAR-10 dataset for 10 classes.
        
        ### Features:
        - Upload single or multiple images.
        - Get real-time predictions with confidence scores.
        - Download results as a CSV file.
        - Visualize confidence scores with bar charts.
        
        Use the sidebar to choose a model and get started!
        """
    )

# Main function to control the navigation
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an Option", ("Welcome", "CIFAR-10 Classification", "MobileNetV2 (ImageNet)"))
    
    if choice == "Welcome":
        welcome_page()
    elif choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10 Classification":
        cifar10_classification()

if __name__ == "__main__":
    main()
