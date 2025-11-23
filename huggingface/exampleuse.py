from gradio_client import Client, handle_file
import os

def analyze_skin_disease(image_path, return_dict=False):
    """
    Analyze skin disease from an image using your Hugging Face model
    
    Args:
        image_path (str): Path to the skin image file
        return_dict (bool): If True, returns dict with success status
        
    Returns:
        str or dict: Prediction result or dict with details
    """
    # Validate file exists
    if not os.path.exists(image_path):
        if return_dict:
            return {"success": False, "error": f"File not found: {image_path}"}
        return f"Error: File not found: {image_path}"
    
    try:
        # Initialize client and make prediction
        client = Client("jianna4/skin-disease-analyser")
        result = client.predict(
            image=handle_file(image_path),
            api_name="/predict_skin"
        )
        
        if return_dict:
            return {
                "success": True,
                "prediction": result,
                "image_path": image_path
            }
        return result
        
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        if return_dict:
            return {"success": False, "error": error_msg, "image_path": image_path}
        return error_msg
    
    # Just get the prediction text
result = analyze_skin_disease("F:\projects\Skin_detection\skin-detetion backend\image.png")
print(result)
