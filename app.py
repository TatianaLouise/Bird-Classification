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
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
     file = FileField("File", validators = [InputRequired()])
     submit = SubmitField("Upload File")

## Set up the path for uploaded images
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = 'static/files'

# Load your saved TensorFlow model

model_path = 'cnn_final.keras'
#model_path = 'test_model.h5'
model = load_model(model_path)

## ADDED ### 
model.make_predict_function()



# # Compile the model manually
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# # Function to preprocess the image
def preprocess_image(image_path):
     img = image.load_img(image_path, target_size = (224, 224))
     img = image.img_to_array(img)/255.0 
     img = np.expand_dims(img, axis=0)
    
     return img 


 ##CHANGED#####
# # # Mapping between bird species index and corresponding names and URLs
# bird_species_info = {
#     # Add more mappings as needed
#     "name": "flamingo", "url": "https://www.allaboutbirds.org/news/flamingos-of-the-altiplano-high-in-the-bolivian-andes/"
#      # Add more mappings as needed
#  }
 
# This was created manually and hard coded, but it is possible to connect to a data base containing the bird_species_labels and the bird_species_info
# Mapping between bird species index and corresponding labels
bird_species_labels = {
    0: "ABBOTTS BABBLER",
    1: "ABBOTTS BOOBY",
    2: "ABYSSINIAN GROUND HORNBILL",
    3: "AFRICAN CROWNED CRANE",
    4: "AFRICAN EMERALD CUCKOO",
    5: "AFRICAN FIREFINCH",
    6: "AFRICAN OYSTER CATCHER",
    7: "AFRICAN PIED HORNBILL",
    8: "AFRICAN PYGMY GOOSE",
    9: "ALBATROSS",
    10: "ALBERTS TOWHEE",
    11: "ALEXANDRINE PARAKEET",
    12: "ALPINE CHOUGH",
    13: "ALTAMIRA YELLOWTHROAT",
    14: "AMERICAN AVOCET",
    15: "AMERICAN BITTERN",
    16: "AMERICAN COOT",
    17: "AMERICAN DIPPER",
    18: "AMERICAN FLAMINGO",
    19: "AMERICAN GOLDFINCH",
    20: "AMERICAN KESTREL",
    21: "AMERICAN PIPIT",
    22: "AMERICAN REDSTART",
    23: "AMERICAN ROBIN",
    24: "AMERICAN WIGEON",
    25: "AMETHYST WOODSTAR",
    26: "ANDEAN GOOSE",
    27: "ANDEAN LAPWING",
    28: "ANDEAN SISKIN",
    29: "ANHINGA",
    30: "ANIANIAU",
    31: "ANNAS HUMMINGBIRD",
    32: "ANTBIRD",
    33: "ANTILLEAN EUPHONIA",
    34: "APAPANE",
    35: "APOSTLEBIRD",
    36: "ARARIPE MANAKIN",
    37: "ASHY STORM PETREL",
    38: "ASHY THRUSHBIRD",
    39: "ASIAN CRESTED IBIS",
    40: "ASIAN DOLLARD BIRD",
    41: "ASIAN GREEN BEE EATER",
    42: "ASIAN OPENBILL STORK",
    43: "AUCKLAND SHAQ",
    44: "AUSTRAL CANASTERO",
    45: "AUSTRALASIAN FIGBIRD",
    46: "AVADAVAT",
    47: "AZARAS SPINETAIL",
    48: "AZURE BREASTED PITTA",
    49: "AZURE JAY",
    50: "AZURE TANAGER",
    51: "AZURE TIT",
    52: "BAIKAL TEAL",
    53: "BALD EAGLE",
    54: "BALD IBIS",
    55: "BALI STARLING",
    56: "BALTIMORE ORIOLE",
    57: "BANANAQUIT",
    58: "BAND TAILED GUAN",
    59: "BANDED BROADBILL",
    60: "BANDED PITA",
    61: "BANDED STILT",
    62: "BAR-TAILED GODWIT",
    63: "BARN OWL",
    64: "BARN SWALLOW",
    65: "BARRED PUFFBIRD",
    66: "BARROWS GOLDENEYE",
    67: "BAY-BREASTED WARBLER",
    68: "BEARDED BARBET",
    69: "BEARDED BELLBIRD",
    70: "BEARDED REEDLING",
    71: "BELTED KINGFISHER",
    72: "BIRD OF PARADISE",
    73: "BLACK AND YELLOW BROADBILL",
    74: "BLACK BAZA",
    75: "BLACK BREASTED PUFFBIRD",
    76: "BLACK COCKATO",
    77: "BLACK FACED SPOONBILL",
    78: "BLACK FRANCOLIN",
    79: "BLACK HEADED CAIQUE",
    80: "BLACK NECKED STILT",
    81: "BLACK SKIMMER",
    82: "BLACK SWAN",
    83: "BLACK TAIL CRAKE",
    84: "BLACK THROATED BUSHTIT",
    85: "BLACK THROATED HUET",
    86: "BLACK THROATED WARBLER",
    87: "BLACK VENTED SHEARWATER",
    88: "BLACK VULTURE",
    89: "BLACK-CAPPED CHICKADEE",
    90: "BLACK-NECKED GREBE",
    91: "BLACK-THROATED SPARROW",
    92: "BLACKBURNIAM WARBLER",
    93: "BLONDE CRESTED WOODPECKER",
    94: "BLOOD PHEASANT",
    95: "BLUE COAU",
    96: "BLUE DACNIS",
    97: "BLUE GRAY GNATCATCHER",
    98: "BLUE GROSBEAK",
    99: "BLUE GROUSE",
    100: "BLUE HERON",
    101: "BLUE MALKOHA",
    102: "BLUE THROATED PIPING GUAN",
    103: "BLUE THROATED TOUCANET",
    104: "BOBOLINK",
    105: "BORNEAN BRISTLEHEAD",
    106: "BORNEAN LEAFBIRD",
    107: "BORNEAN PHEASANT",
    108: "BRANDT CORMARANT",
    109: "BREWERS BLACKBIRD",
    110: "BROWN CREPPER",
    111: "BROWN HEADED COWBIRD",
    112: "BROWN NOODY",
    113: "BROWN THRASHER",
    114: "BUFFLEHEAD",
    115: "BULWERS PHEASANT",
    116: "BURCHELLS COURSER",
    117: "BUSH TURKEY"
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

# # Mapping between bird species index and corresponding labels
# bird_species_labels = {
#     0: "flamingo",
#     1: "mosquito_net",
#     2: "eagle"
#     # Add more mappings as needed for other bird species
# }

# # Mapping between bird species labels and corresponding URLs
# bird_species_info = {
#     "flamingo": {
#         "name": "flamingo",
#         "url": "https://www.allaboutbirds.org/news/flamingos-of-the-altiplano-high-in-the-bolivian-andes/"
#     },
#     "mosquito_net": {
#         "name": "mosquito_net",
#         "url": "https://www.amazon.com/mosquito-net/s?k=mosquito+net"
#     },
#     "eagle": {
#         "name": "eagle",
#         "url": "https://www.example.com/eagle"
#     }
#     # Add more mappings as needed for other bird species
# }




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
        #############p = preprocess_image(file_path)
        ###CHANGED#########
        image = preprocess_image(file_path)

        print(image.shape)
        predictions = model.predict(image)
        print('tatiana')
        print(predictions)
        print(predictions.shape)


        
        predicted_class_idx = np.argmax(predictions[0])
        print('predicted_class_idx:', predicted_class_idx)


        
        bird_species_label = bird_species_labels.get(predicted_class_idx, "Unknown")
        bird_info = bird_species_info.get(bird_species_label)



        # top_3 = decode_predictions(predictions, top=3)
        
        
        # predicted_class_label = top_3[0][0][1]
        # print('top_3', top_3)
        # print(predicted_class_label)

                        ###bird_species_name = bird_species_labels.get(predicted_class_label)
       #############bird_species_name = 'flamingo'
        # Get the predicted bird species name using the index
        
        #bird_species_name = bird_species_labels.get(predicted_class_idx, "Unknown")
        #print('bird_species_name:', bird_species_name)  # Add this line to check the bird species name
        # Get the bird info directly from the bird_species_info dictionary
        
        # bird_info = bird_species_info.get(predicted_class_label)
                         #bird_info = bird_species_info.get(bird_species_name)
        print('bird_info:', bird_info)  # Print bird_info for debugging purposes
        os.remove(file_path)
        if bird_info:
            print('Redirecting to:', bird_info.get('url', '/'))
            return redirect(bird_info.get('url', '/'))  # Redirect to the URL, defaulting to home page
        
        else:
            return "Species not found"

##### ADDED ######
        
        #predicted_labels = decode_predictions(prediction, top=3)[0]  # Extract top 3 predicted labels
        #predicted_label = predicted_labels[0][1]  # Get the label of the top predicted class
        # Example: predicted_label = 'flamingo'
             # Get the corresponding bird species info
        







#####CHANGED ######
        #bird_info = bird_species_info.get(predicted_class_idx)
    
    return render_template('demo.html', form=form)


    #  form = UploadFileForm()
    #  if form.validate_on_submit():
    #       file = form.file.data
    #       file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
    #       return "File has been uploaded"
    #  return render_template('demo.html', form=form)









# def identify_bird(image_path, model):
#      image = preprocess_image(image_path)
#      prediction = model.predict(image)
#      predicted_class_idx = np.argmax(prediction[0])
#      return predicted_class_idx



if __name__ == '__main__':
#      Ensure the UPLOAD_FOLDER exists
     #os.makedirs(UPLOAD_FOLDER, exist_ok=True)
     app.run(debug=True)




