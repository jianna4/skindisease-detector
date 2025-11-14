#imports
import gradio as gr #this is what we will use for huggingface space,its our UI
import tensorflow as tf # WE  WILL USE TENSORFLOW TO LOAD OUR MODEL
import numpy as np # to cnvert images into arrrays
from PIL import Image # to process images

#Load the model
model = tf.keras.models.load_model('skin_disease_model.h5') #huggingface will find this file in your repository

#ddefine the class names

CLASS_NAMES = [Acne,Carcinoma ,Eczema, Keratosis ,Milia ,Rosacea] #Just in the order of your model output dataset

#define input size
INPUT_SIZE = (224,224) #input size of your model

# now the prediction function
def predict_skin(image):
    # Preprocess the image
    image = image.resize(INPUT_SIZE)
    image = np.array(image) / 255.0  # Normalize to [0, 1] to make it a smaller number 
    image = np.expand_dims(image, axis=0)  # Add batch dimension Because the model architecture itself was built around batches from the beginning — not just during training.

    # Make prediction
    predictions = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    return predicted_class