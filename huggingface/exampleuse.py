import requests
from io import BytesIO

def get_skin_prediction(image_input):
    """
    Sends an image to the skin classifier API and returns the predicted class.

    Parameters:
        image_input: str (file path) or bytes (raw image data)

    Returns:
        str: predicted class (e.g., 'Acne', 'Eczema', etc.)
    """
    API_URL = "https://huggingface.co/spaces/jianna4/cnn_new_disease_detector/predict/"  # the API endpoint

    # Convert the input into a file-like object
    if isinstance(image_input, str):
        file_to_send = open(image_input, "rb")
    elif isinstance(image_input, bytes):
        file_to_send = BytesIO(image_input)
    else:
        raise ValueError("image_input must be a file path (str) or bytes")
    
    # Send the POST request
    response = requests.post(API_URL, files={"file": file_to_send})
    
    # Close the file if opened from disk
    if isinstance(image_input, str):
        file_to_send.close()
    
    # Return prediction or raise error
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        raise RuntimeError(f"API request failed: {response.status_code}")
