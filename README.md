🩺 Skin Disease Classifier
A deep learning web application that classifies skin diseases from images using a Convolutional Neural Network (CNN). The app can identify 6 different skin conditions.



🌟 Features
Multi-class Classification: Identifies 6 skin conditions:

🎯 Acne

🎯 Carcinoma

🎯 Eczema

🎯 Keratosis

🎯 Milia

🎯 Rosacea

User-Friendly Interface: Simple drag-and-drop image upload

Real-time Predictions: Instant results with confidence scores

Web Deployment: Accessible via Hugging Face Spaces

🚀 Live Demo
https://jianna4-skin-disease-analyser.hf.space/

🛠️ Technical Stack
Framework: TensorFlow/Keras

Model Architecture: CNN with 3 Conv2D layers + Dense layers

Frontend: Gradio

Image Processing: PIL, NumPy

Deployment: Hugging Face Spaces

📁 Project Structure
text
skin-disease-classifier/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── your_modelnn.h5       # Trained model file
├── README.md             # This file

🧠 Model Architecture
python
model = tf.keras.models.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='softmax')
])
🎯 How to Use
Visit the app through the Hugging Face Space link

Upload an image of a skin condition

Click Submit or wait for auto-prediction

View results showing the predicted condition and confidence

Supported Image Formats
JPEG, JPG, PNG

Recommended size: 224x224 pixels

Color images (RGB)

🔧 Local Development
Prerequisites
Python 3.8+

pip

Installation
Clone the repository

bash
git clone https://huggingface.co/spaces/your-username/skin-disease-classifier
cd skin-disease-classifier
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
python app.py
Open your browser to http://localhost:7860

Requirements
txt
tensorflow-cpu==2.12.0
gradio==3.50.2
pillow>=9.0.0
numpy>=1.21.0
📊 Dataset & Training
The model was trained on an augmented skin conditions dataset with the following structure:

text
Skin_Conditions/
├── Acne/
├── Carcinoma/
├── Eczema/
├── Keratosis/
├── Milia/
└── Rosacea/
Training Details:

Image Size: 224x224 pixels

Batch Size: 32

Optimizer: Adam (learning_rate=1e-5)

Loss: Sparse Categorical Crossentropy

Regularization: L2 regularization applied

⚠️ Important Disclaimer
This application is for educational and demonstration purposes only.

Not Medical Advice: Predictions should not be used for medical diagnosis

Consult Professionals: Always consult healthcare professionals for medical concerns

Experimental: Model accuracy may vary with real-world data

Limitations: Performance depends on training data quality and diversity

🤝 Contributing
Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

Areas for Improvement
Add more skin conditions

Improve model accuracy

Add multi-language support

Include explainable AI features

Mobile app development

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Dataset providers for skin condition images
/kaggle/input/augmented-skin-conditions-image-dataset/Skin_Conditions
Hugging Face for the free hosting platform

TensorFlow/Keras team for the deep learning framework

Gradio for the simple web interface framework

The model has an endpoint ready to use as:https://huggingface.co/spaces/jianna4/cnn_new_disease_detector/predict
kaggle dataset:https://www.kaggle.com/datasets/syedalinaqvi/augmented-skin-conditions-image-dataset/
kaggle notebook:https://www.kaggle.com/code/joanwachuka/skindisease2/notebook?scriptVersionId=281234154

you can aslo check out the example of how to setup and use the API in the file exampleuse.py
