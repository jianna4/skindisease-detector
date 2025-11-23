# imports
import gradio as gr  # this is what we will use for huggingface space, its our UI
import tensorflow as tf  # WE WILL USE TENSORFLOW TO LOAD OUR MODEL
import numpy as np  # to convert images into arrays
from PIL import Image  # to process images
import uvicorn  # to deploy the fastapi app
from fastapi import FastAPI, UploadFile, File
from io import BytesIO

# Load the model
model = tf.keras.models.load_model('finalmodel.keras')  # huggingface will find this file in your repository

# define the class names
CLASS_NAMES = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']  # Just in the order of your model output dataset

# define input size
INPUT_SIZE = (224, 224)  # input size of your model

# The treatments
TREATMENTS = {
    'Acne': "Use topical benzoyl peroxide or salicylic acid. Keep skin clean and avoid picking.",
    'Carcinoma': "Consult a dermatologist immediately. Early diagnosis is critical.",
    'Eczema': "Apply moisturizers regularly and avoid irritants. Use steroid creams if prescribed.",
    'Keratosis': "Consult a dermatologist. Treatment may involve cryotherapy or topical creams.",
    'Milia': "Usually harmless, may remove with gentle exfoliation or see a dermatologist.",
    'Rosacea': "Avoid triggers like spicy food or alcohol. Use gentle skin care and prescribed treatments."
}


# now the prediction function
def predict_skin(image):
    # Preprocess the image
    image = image.resize(INPUT_SIZE)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    # Get treatment
    treatment = TREATMENTS.get(predicted_class, "No treatment information available.")
    return f"Prediction: {predicted_class}\nSuggested Treatment: {treatment}"


# fastapi setup
api = FastAPI(title="Skin Disease Classifier API")

@api.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    prediction = predict_skin(image)
    return {"prediction": prediction}


# 🔥 UPGRADED GRADIO INTERFACE (only this part changed)
interface = gr.Interface(
    fn=predict_skin,  # function to call
    inputs=gr.Image(type="pil", label="Upload Skin Image",height=300,sources=["upload", "webcam"] ),  # input
    outputs=gr.Textbox(
        label="Prediction",
        lines=5,            # number of lines visible
        max_lines=10,       # maximum lines if text wraps
        placeholder="Prediction will appear here..."
    ),
    title="Skin Disease Classifier",
    description="Upload an image of a skin area and the model will predict the disease category as either Acne, Carcinoma, Eczema, Keratosis, Milia or Rosacea."
)
interface.launch(share=True) #launch the app and create public link

# Mount the Gradio app inside FastAPI
api = gr.mount_gradio_app(api, interface, path="/")

# Only needed for local run (ignored on HF Spaces)
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=7860)
