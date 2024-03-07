import os
import numpy as np
import cv2
import pdfkit
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, make_response
from werkzeug.utils import secure_filename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
app = Flask(__name__)

model = load_model(('Model_Brain_Tumor.h5'), custom_objects={'KerasLayer': hub.KerasLayer})

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

#Model page route
@app.route('/model')
def about():
    return render_template('model.html')

#Model page route
@app.route('/contact')
def contact():
    return render_template('contact.html')

def model_predict(img_path, loaded_model):
   # Read and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 224, 224, 3)

    # Make prediction
    predictions = loaded_model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    labels = ['pituitary_tumor', 'no_tumor',  'meningioma_tumor', 'glioma_tumor']

    # Get the predicted class label
    predicted_class_label = labels[predicted_class_index]

    # Print the results
    print("Predicted class index:", predicted_class_index)
    print("Predicted class label:", predicted_class_label)

    return predicted_class_label

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds

        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)