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
