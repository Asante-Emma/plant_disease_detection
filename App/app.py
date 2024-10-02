from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('../models/plant_disease_model.keras')

# Path to the class labels
classes = ['Tomato_healthy', 'Potato___Early_blight', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
           'Tomato_Early_blight', 'Tomato__Target_Spot', 'Potato___Late_blight', 'Tomato_Leaf_Mold', 
           'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Septoria_leaf_spot', 'Tomato__Tomato_mosaic_virus',
           'Pepper__bell___Bacterial_spot', 'Tomato_Bacterial_spot', 'Tomato_Late_blight', 
           'Pepper__bell___healthy', 'Potato___healthy']

# Prediction function
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return classes[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['image']
        if file:
            # Save the image
            image_path = 'static/' + file.filename
            file.save(image_path)

            # Predict the image class
            prediction = predict_image(image_path)
            return render_template('result.html', prediction=prediction, image_path=image_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
