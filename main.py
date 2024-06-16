# Error Might Occur:
# Drivr API enable and json file  is used
# Drive api integration link : https://youtu.be/tamT_iGoZDQ?si=n5CP-uJrnKbvosNm
# Complete project running in colab:https://colab.research.google.com/drive/1sr-rnQKVUhVrtWgRcnuJQvuoHHcDf4hu?usp=sharing


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
import io
from PIL import Image
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# Google Drive Creds
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = '/content/drive/MyDrive/Image_classification/image-classification-426509-6d6247c00caa.json'  # Drivr API Json File | Ensure this path is correct 
PARENT_FOLDER_ID = '1Rbakr5RDFVOO5IYHNta4ydg-NTlDHEJY' #saving uploaded img in Folder in Drive 

# Google Drive Function to upload images
def authenticate():
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

def upload_photo(file_path):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': 'img.jpg',
        'parents': [PARENT_FOLDER_ID]
    }

    media = MediaFileUpload(file_path, mimetype='image/jpeg')

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    return file.get('id')

def list_files_in_folder(folder_id):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    return files

def get_image_from_drive(file_id):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    img = Image.open(fh)
    return img

# Load the pre-trained VGG19 model
model = VGG19(weights='imagenet', include_top=True, classifier_activation='softmax')

def classify_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict the class of the image
    preds = model.predict(x)

    # Decode the prediction
    label = decode_predictions(preds, top=1)[0][0][1]
    score = decode_predictions(preds, top=1)[0][0][2]

    return label, score

# Define the about page
def about_page():
    st.title("About This App")
    st.write("""
    ## Image Classification with VGG19

    This app allows you to upload an image and classify it using the pre-trained VGG19 model. 
    The VGG19 model is a deep convolutional neural network that has been trained on the ImageNet dataset, 
    which contains over 14 million labeled images across 1,000 different classes.

    ### How It Works
    1. **Upload an Image**: Use the file uploader to select an image in JPG format.
    2. **Image Processing**: The uploaded image is preprocessed to match the input requirements of the VGG19 model.
    3. **Prediction**: The VGG19 model predicts the class of the image, and the result is displayed along with the confidence score.
    4. **Visualization**: A bar chart is generated to visualize the confidence score for the predicted class.
    5. **Upload to Google Drive**: The uploaded image is also saved to a specified folder in your Google Drive.

    ### Technologies Used
    - **Streamlit**: For creating the web application.
    - **TensorFlow & Keras**: For loading the VGG19 model and making predictions.
    - **Google Drive API**: For uploading the images to Google Drive.
    - **Matplotlib**: For visualizing the prediction confidence score.

    ### Credits
    This app was developed using various open-source libraries and APIs. Special thanks to the developers and maintainers of these tools and libraries.
    """)
    st.write("### Contact")
    st.write("For any inquiries or feedback, please contact [Your Name](mailto:your.email@example.com).")

# Main app logic
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Home", "About", "View Images"])

if page == "Home":
    st.title("Image Classification with VGG19")
    st.write("Upload an image to classify it using the VGG19 model.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Classify the image
        predicted_label, confidence_score = classify_image("temp_image.jpg")
        
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write(f"The image contains: {predicted_label} with confidence: {confidence_score:.2f}")

        # Upload the image to Google Drive
        file_id = upload_photo("temp_image.jpg")
        st.write(f"Uploaded image to Google Drive with file ID: {file_id}")
        
        # Plot the confidence score
        fig, ax = plt.subplots()
        ax.bar(predicted_label, confidence_score, color='blue')
        ax.set_title('Prediction Confidence')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 1)
        
        st.pyplot(fig)
elif page == "About":
    about_page()
elif page == "View Images":
    st.title("View Images from Google Drive Folder")

    files = list_files_in_folder(PARENT_FOLDER_ID)
    if files:
        for file in files:
            st.write(f"Image: {file['name']}")
            img = get_image_from_drive(file['id'])
            st.image(img, caption=file['name'], use_column_width=True)
    else:
        st.write("No images found in the folder.")