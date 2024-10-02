import cv2
import numpy as np
import tensorflow as tf

# Load the trained plant disease classification model
disease_model = tf.keras.models.load_model('../models/plant_disease_model.keras')

# Class labels
classes = ['Tomato_healthy', 'Potato___Early_blight', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
           'Tomato_Early_blight', 'Tomato__Target_Spot', 'Potato___Late_blight', 'Tomato_Leaf_Mold', 
           'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Septoria_leaf_spot', 'Tomato__Tomato_mosaic_virus',
           'Pepper__bell___Bacterial_spot', 'Tomato_Bacterial_spot', 'Tomato_Late_blight', 
           'Pepper__bell___healthy', 'Potato___healthy']

# Function to preprocess and predict disease from the cropped leaf image
def predict_disease(cropped_img):
    img = cv2.resize(cropped_img, (128, 128))  # Resize to match model input
    img_array = np.expand_dims(img / 255.0, axis=0)  # Normalize and expand dims
    predictions = disease_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return classes[predicted_class]

# Function to detect plants (leaves) and predict diseases using contour detection
def detect_and_classify(image_path):
    # Load the input image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color range for green leaves in HSV space
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Create a mask to extract green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Use the mask to segment the leaves
    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    # Convert the segmented image to grayscale for contour detection
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    
    # Find contours from the segmented leaf regions
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to keep track of detected bounding boxes
    boxes = []

    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    # Sort bounding boxes based on the y-coordinate to avoid label overlap
    boxes = sorted(boxes, key=lambda b: b[1])

    for (x, y, w, h) in boxes:
        crop_img = img[y:y+h, x:x+w]  # Crop the detected leaf

        if crop_img.size > 0:
            # Predict the disease for the cropped image
            disease = predict_disease(crop_img)
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label with background to avoid overlapping
            label = f"{disease}"
            font_scale = 0.7
            font_thickness = 2
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            label_background = (0, 0, 0)  # Black background
            
            # Label position
            label_x = x
            label_y = max(y - 10, 20)  # Shift label upwards and ensure it's within image bounds

            # Draw background rectangle
            cv2.rectangle(img, (label_x, label_y - label_size[1] - 10), 
                          (label_x + label_size[0], label_y), label_background, -1)
            # Draw the label
            cv2.putText(img, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # Display the image with bounding boxes and disease labels
    cv2.imshow('Detected Plants and Diseases', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path1 = '../plant_images/68a9df4a-f803-4cf1-bfce-1a637a2befcc___RS_Early.B 7654.JPG'
    image_path2 = '../plant_images/1b150c8d-ab5d-43b5-a61d-c3fb3a8ffd54___GH_HL Leaf 321.JPG'
    image_path3 = '../plant_images/8f634ae3-010a-4a08-aae9-2aa43bc73a0a___RS_LB 4069.JPG'
    detect_and_classify(image_path2)
