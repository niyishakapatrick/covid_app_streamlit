import streamlit as st
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


header = st.container()
model_inference = st.container()
features = st.container()
col1, col2 = st.columns(2)


with header:
    # Add custom CSS styling
    st.markdown(
        """
        <style>
        .css-1aumxhk {
            background-image: linear-gradient(to right, rgba(255, 0, 0, 0.5), rgba(0, 0, 255, 0.5), rgba(0, 255, 0, 0.5));
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Render the navbar
    st.markdown(
        """
        <div class="css-1aumxhk">
        <h2 style="color: white;">Detecting COVID-19 through chest X-rays</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with model_inference:
    
    preprocess = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.445, 0.445, 0.445], std=[0.269, 0.269, 0.269])
    ])

    # Load the model's parameters, mapping them to the CPU if necessary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xray_detection_model.pth')
    model1 = EfficientNet.from_pretrained('efficientnet-b2', num_classes=2)
    model1.load_state_dict(torch.load(model_path1, map_location=device)['model_state_dict'])
    model1.to(device)
    model1.eval()


    # Load the model's parameters, mapping them to the CPU if necessary
    model_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'covid_detection_model.pth')
    model2 = EfficientNet.from_pretrained('efficientnet-b2', num_classes=2)
    model2.load_state_dict(torch.load(model_path2, map_location=device)['model_state_dict'])
    model2.to(device)
    model2.eval()

    def check_xray(image):
        # Apply the transformations to the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Perform inference on the image
        with torch.no_grad():
            input_batch = input_batch.to(device)
            output = model1(input_batch)
        
        # Get the predicted class
        _, predicted_idx = torch.max(output, 1)
        predicted_label = predicted_idx.item()
        
        # Define the class labels
        classes = ['0_others', '1_chest_xray']
        
        # Get the predicted label and probabilities
        predicted_class = classes[predicted_label]
        probabilities = torch.softmax(output, dim=1)
        prob_chest_xray = probabilities[0][0].item()
        prob_not_chest_xray = probabilities[0][1].item()
        
        return predicted_class, prob_chest_xray, prob_not_chest_xray

    def perform_inference(image):
        # Apply the transformations to the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Perform inference on the image
        with torch.no_grad():
            input_batch = input_batch.to(device)
            output = model2(input_batch)
        
        # Get the predicted class
        _, predicted_idx = torch.max(output, 1)
        predicted_label = predicted_idx.item()
        
        # Define the class labels
        classes = ['COVID-19', 'Normal']
        
        # Get the predicted label and probabilities
        predicted_class = classes[predicted_label]
        probabilities = torch.softmax(output, dim=1)
        prob_covid = probabilities[0][0].item()
        prob_normal = probabilities[0][1].item()
        
        return predicted_class, prob_covid, prob_normal

    def upload_image():
        # Upload and display the image
        uploaded_image = st.file_uploader("Upload an image (chest X-ray image only)", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            # Convert grayscale image to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            predicted_class_xray, prob_chest_xray, prob_not_chest_xray = check_xray(image)

            # If the image is detected as a chest X-ray, perform COVID-19 inference
            if predicted_class_xray == '1_chest_xray':
                st.write("The uploaded image is  a chest X-ray.")
                # Perform inference on the image
                predicted_class_covid, prob_covid, prob_normal = perform_inference(image)

                # Get image dimensions
                width, height = image.size

                # Display the image and inference results
                st.image(image, caption="Uploaded Image", width=300)
                # Convert probabilities to percentages
                prob_covid_percent = prob_covid * 100
                prob_normal_percent = prob_normal * 100

                # Plot the probabilities
                labels = ['COVID-19', 'Normal']
                probabilities = [prob_covid_percent, prob_normal_percent]
                colors = ['red', 'blue']

                # Write details to CSV file
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data = {
                    'Image Name': [uploaded_image.name],
                    'Chest X-ray': [True],
                    'COVID-19 prob.': [prob_covid_percent],
                    'Normal prob.': [prob_normal_percent],
                    'Timestamp': [current_time]
                }
                df = pd.DataFrame(data)
                df.to_csv('detection_results.csv', mode='a', header=False, index=False)

                fig, ax = plt.subplots()
                ax.barh(labels, probabilities, color=colors)
                ax.set_xlim(0, 100)  # Set x-axis limit from 0 to 100 (percentage range)
                ax.set_xlabel('Probability (%)')

                # Display the number values on the plot
                for i, v in enumerate(probabilities):
                    ax.text(v + 1, i, str(round(v, 2)), color='black', va='center')

                # Display the image and the plot
                st.pyplot(fig)
            else:
                st.write("The uploaded image is not a chest X-ray.")

    # Call the function to run the Streamlit app
    upload_image()


with features:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Differentiate COVID-19 from normal cases in X-ray images using ML")
    st.markdown("""Detecting COVID-19 from chest X-ray images involves applying a trained machine learning model to analyze and categorize the images for signs of the infection. Deep learning methods enhance accuracy and speed in diagnosing COVID-19 cases, aiding healthcare workers. However, it's crucial to recognize that model outputs support medical professionals rather than replace clinical judgment, as X-rays alone may not conclusively diagnose COVID-19. Seeking medical expertise and additional tests remain essential for accurate diagnosis and patient treatment.""")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Salient features for COVID-19 detection")
    st.markdown("COVID-19 chest X-ray features: Ground-glass opacities, consolidation, bilateral involvement, peripheral distribution, crazy paving pattern. These features can also be present in respiratory conditions other than COVID-19. Confirming COVID-19 requires additional tests and evaluation by healthcare professionals.")

with col1:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.image('static/covid19.png', caption='COVID-19', use_column_width=True)

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.image('static/normal.jpg', caption='Normal', use_column_width=True)
   


