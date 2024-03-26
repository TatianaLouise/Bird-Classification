# Bird-Classification with CNN Model 

## Data Set
Our data can be found here: https://www.kaggle.com/datasets/gpiosenka/100-bird-species 

## Project Objectives 
- Leverage a CNN deep learning model for bird classification.
- Develop a machine learning model utilizing a dataset comprising 118 bird species.
- Classify birds into respective species using image inputs.
- Display practical applications of the model.


## Creation Process:

1. Data Exploration and Preparation:
Explored the dataset directory structure to understand the organization of the data.
Loaded the class names by extracting directory names from the training dataset.
Visualized random images from different classes to get an understanding of the data.

2. Data Preprocessing:
Loaded and processed images, including resizing and rescaling.
Created data generators using ImageDataGenerator to load images from directories and apply data augmentation techniques like rescaling.

3. Model Architecture:
Utilized the pre-trained model InceptionResNetV2 as the base model for feature extraction.
Froze the base model's layers to prevent them from being trained again.
Added a global average pooling layer to reduce the spatial dimensions of the features.
Added a dense output layer with softmax activation for classification.

4. Model Compilation and Training:
Compiled the model with the Adam optimizer.
Fit the model for training, specifying the training data and defined number of epochs.
Evaluated the model performance on the test data.
Saved the trained model in the native Keras format for future use.

5. Model Loading and Prediction:
Loaded the saved model using TensorFlow/Keras.
Defined a function to predict and plot a single image using the loaded model. 

6. Flask application: 
This model connects to a flask web application with a user friendly interface. The user has to option to upload bird image inputs. The model will classify the bird species, and link the user to the Cornell Lab of Ornithology (https://www.birds.cornell.edu/home/), an educational resource for more detailed bird information. If the model is unable to identify the species of bird from the input, "species not found" will be displayed to the user.  

Overall, this CNN model was constructed by leveraging transfer learning with a pre-trained InceptionResNetV2 model, followed by fine-tuning on the specific dataset. Data augmentation techniques were applied during training to improve model generalization. The final model was saved for deployment and future use.


## Presentation 
Link to Presentation: https://www.canva.com/design/DAF_ipxWAdc/BkWT2ORVnH0tfE6ClK59pQ/edit?utm_content=DAF_ipxWAdc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 


This project was developed by Josh Mares, Madison Hamby, Matthew Garcia, and Tatiana Oropel. 
- [Josh Mares] - https://github.com/Xelven001 
- [Madison Hamby] - https://github.com/madisonhamby 
- [Matthew Garcia] - https://github.com/matthewjgarcia 
- [Tatiana Oropel] - https://github.com/TatianaLouise 

For detailed information, refer to the project code and documentation.

