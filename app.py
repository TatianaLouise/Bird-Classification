import os
from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input,
    decode_predictions
)
import numpy as np
from PIL import Image

app = Flask(__name__)

## Set up the path for uploaded images
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
     file = FileField("File", validators = [InputRequired()])
     submit = SubmitField("Upload File")

# Load saved TensorFlow model

model_path = 'cnn_final.keras'
model = load_model(model_path)

## ADDED to predict images with the model ### 
model.make_predict_function()

# # Compile the model manually
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
## Define function to preprocess the image
def preprocess_image(image_path):
     img = image.load_img(image_path, target_size = (224, 224))
     img = image.img_to_array(img)/255.0 
     img = np.expand_dims(img, axis=0)
    
     return img 
 
# This was created manually and hard coded, but it is possible to connect to a data base containing the bird_species_labels and the bird_species_info
# Mapping between bird species index and corresponding labels
bird_species_labels = {
    2: "ABYSSINIAN GROUND HORNBILL",
    3: "AFRICAN CROWNED CRANE",
    8: "AFRICAN PYGMY GOOSE",
    74: "BLACK BAZA",
    93: "BLONDE CRESTED WOODPECKER",
    112: "BROWN NODDY"
    # Add more mappings as needed for other bird species
}

# Mapping between bird species labels and corresponding URLs
bird_species_info = {
    "ABYSSINIAN GROUND HORNBILL": {
        "name": "ABYSSINIAN GROUND HORNBILL",
        "url": "https://ebird.org/species/noghor1"
    },
    "AFRICAN CROWNED CRANE": {
        "name": "AFRICAN CROWNED CRANE",
        "url": "https://ebird.org/species/grccra1"
    },
    "AFRICAN PYGMY GOOSE": {
        "name": "AFRICAN PYGMY GOOSE",
        "url": "https://ebird.org/species/afrpyg1"
    },
    "BLACK BAZA": {
        "name": "BLACK BAZA",
        "url": "https://ebird.org/species/blabaz1"
    },
    "BLONDE CRESTED WOODPECKER": {
        "name": "BLONDE CRESTED WOODPECKER",
        "url": "https://ebird.org/species/blcwoo4"
    },
    "BROWN NODDY": {
        "name": "BROWN NODDY",
        "url": "https://ebird.org/species/brnnod"
    },
    # Add more mappings as needed for other bird species
}


# Routes
@app.route('/', methods = ['GET', 'POST'])
def home_page():
     return render_template('Webpage.html')

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Process the image using the model
        image = preprocess_image(file_path)

        print(image.shape)
        predictions = model.predict(image)

        # Print results for each step to make sure model is running smoothly
        print('hello')
        print(predictions)
        print(predictions.shape)

        predicted_class_idx = np.argmax(predictions[0])
        print('predicted_class_idx:', predicted_class_idx)

        # Get the predicted bird species name using the index
        bird_species_label = bird_species_labels.get(predicted_class_idx, "Unknown")
        bird_info = bird_species_info.get(bird_species_label)
        
        # Get the bird info directly from the bird_species_info dictionary

        print('bird_info:', bird_info)  # Print bird_info for debugging purposes
        os.remove(file_path)
        if bird_info:
            print('Redirecting to:', bird_info.get('url', '/'))
            return redirect(bird_info.get('url', '/'))  # Redirect to the URL, defaulting to home page
        else:
            return "Species not found"
    
    return render_template('demo.html', form=form)

if __name__ == '__main__':
     app.run(debug=True)




