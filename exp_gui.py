import cv2
import numpy as np
import streamlit as st
from keras.models import load_model

# Load the emotion model
model_path = "fer2013_mini_XCEPTION.107-0.66.hdf5"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model(model_path, compile=False)

# Define a dictionary to map expressions to emojis
emoji_mapping = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Sad': '😢',
    'Surprise': '😲',
    'Neutral': '😐'
}

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect face
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces, gray

# Function to predict expression
def predict_expression(roi_gray):
    roi_gray = cv2.resize(roi_gray, (64, 64))
    roi_gray = roi_gray.astype('float')
    roi_gray /= 255.0
    roi_gray = np.expand_dims(roi_gray, axis=0)
    roi_gray = np.expand_dims(roi_gray, axis=-1)
    prediction = model.predict(roi_gray)
    max_index = np.argmax(prediction)
    emotion_label = emotion_labels[max_index]
    confidence_score = prediction[0][max_index]
    return emotion_label, confidence_score

# Streamlit UI
st.title('Facial Expression Recognition')

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

if st.button('Capture Photo'):
    ret, frame = cap.read()
    if ret:
        faces, gray = detect_face(frame)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Assuming we capture only one face
            roi_gray = gray[y:y + h, x:x + w]
            emotion_label, confidence_score = predict_expression(roi_gray)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion_label} ({confidence_score:.2f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            st.image(frame, channels='BGR', use_column_width=True)
            emoji = emoji_mapping.get(emotion_label, '😐')  # Default to a neutral emoji
            st.write(f'Expression: {emotion_label} {emoji}')
            st.write(f'Confidence Score: {confidence_score:.2f}')
        else:
            st.warning('No face detected in the captured photo.')
    else:
        st.error('Failed to capture a photo.')

# Release the video capture
cap.release()
