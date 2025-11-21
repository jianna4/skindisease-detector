#imports
import gradio as gr #this is what we will use for huggingface space,its our UI
import tensorflow as tf # WE  WILL USE TENSORFLOW TO LOAD OUR MODEL
import numpy as np # to cnvert images into arrrays
from PIL import Image # to process images
import uvicorn  #to deploy the fasapi app
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
#Load the model
model = tf.keras.models.load_model('vweryverynew.keras') #huggingface will find this file in your repository

#ddefine the class names

CLASS_NAMES = ['Acne','Carcinoma' ,'Eczema', 'Keratosis' ,'Milia' ,'Rosacea'] #Just in the order of your model output dataset

#define input size
INPUT_SIZE = (224,224) #input size of your model

# now the prediction function
def predict_skin(image):
    # Preprocess the image
    image = image.resize(INPUT_SIZE)
    image = np.array(image) / 255.0  # Normalize to [0, 1] to make it a smaller number 
    image = np.expand_dims(image, axis=0)  # Add batch dimension Because the model architecture itself was built around batches from the beginning — not just during training.
    #this will make the image a batch of 1 now the array looks something like [1,224,224,3] instead of [224,224,3]


    # Make prediction
    predictions = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    #argmax, argument of themaximum , gives the index of the highest value in the predictions array eg if predictions = [0.1,0.7,0.2] then argmax will return 1 because 0.7 is the highest value at index 1,whichwill be Carcinoma in our case
    return predicted_class

#fastapi setup

api = FastAPI(title="Skin Disease Classifier API")
@api.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
     # Convert raw bytes to a PIL Image object
    # BytesIO converts bytes into a file-like object so PIL can read it
    # '.convert("RGB")' ensures the image has 3 color channels (Red, Green, Blue)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    
    prediction = predict_skin(image)
    # Return a JSON response containing the prediction
    # FastAPI automatically converts the Python dict into JSON
    return {"prediction": prediction}
#gradio interface

interface = gr.Interface(
    fn=predict_skin,#the function to call
    inputs=gr.Image(type="pil", label="Upload Skin Image"), # type of input to accept ...this it the python image library
    outputs=gr.Text(label="Prediction"),#output label
    title="Skin Disease Classifier", # title of the app
    description="Upload an image of a skin area and the model will predict the disease category as either Acne,Carcinoma ,Eczema, Keratosis ,Milia or Rosacea." #description of the app for users
)

interface.launch(share=True) #launch the app and create public link

# Mount the Gradio app inside FastAPI
#this is very important if the inerface is not mounted the fastapi app will not work on huggingface space
api = gr.mount_gradio_app(api, interface, path="/")

# Only needed for local run (ignored on HF Spaces)
# this will run the fastapi app on local machine and can be accessed on localhost:7860
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=7860)