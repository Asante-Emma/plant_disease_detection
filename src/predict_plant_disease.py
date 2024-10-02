import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

# Load the saved model
model = tf.keras.models.load_model('../models/plant_disease_model.keras')

# Path to the class labels
classes = ['Tomato_healthy', 'Potato___Early_blight', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_Early_blight', 
           'Tomato__Target_Spot', 'Potato___Late_blight', 'Tomato_Leaf_Mold', 
           'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Septoria_leaf_spot', 'Tomato__Tomato_mosaic_virus',
           'Pepper__bell___Bacterial_spot', 'Tomato_Bacterial_spot', 'Tomato_Late_blight', 
           'Pepper__bell___healthy', 'Potato___healthy']

# Function to preprocess image and predict
def predict_image(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=(128, 128))
    
    # Convert the image to array and normalize
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Print the result
    #print(f"Predicted Class: {classes[predicted_class]}")
    return classes[predicted_class]

if __name__ == '__main__':
    image_path = '../plant_images/2d34f331-50b9-4d90-851c-4b0d2ebc4f31___JR_HL 8724.JPG'
    print(predict_image(image_path))