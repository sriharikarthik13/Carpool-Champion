import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import easyocr


from tensorflow.keras.preprocessing.image import load_img, img_to_array


model = tf.keras.models.load_model('../models/obj_detect.keras')
print('model loaded sucessfully')

# create pipeline
def anpr_extract(path):
    #Read Image
    img = load_img(path)
    #Covert to 8 bit array
    img = np.array(img,dtype=np.uint8)
    #Covert to required shape
    img1 = load_img(path,target_size=(224,224))
    #normalized array of the image
    norm_img = img_to_array(img1)/255.0
    height,width,depth = img.shape
    val_array = norm_img.reshape(1,224,224,3)
     #Predictions
    preds = model.predict(val_array)
    #Denormalize the values
    norm_values_d = np.array([width,width,height,height])
    preds = preds * norm_values_d
    preds = preds.astype(np.int32)
    #Draw the boundary box
    xmin,xmax,ymin,ymax = preds[0]
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)

    #LET US PERFORM OCR TEXT EXTRACTION
    imga = np.array(load_img(path))

    required_area = imga[ymin:ymax,xmin:xmax]


    reader = easyocr.Reader(['en'])
    result = reader.readtext(required_area)
    if result != []:
        result = result[0][1]
    
    return img,required_area,result


