from flask import Flask, request, render_template,make_response
import json
import base64
import tensorflow as tf
from tensorflow.keras.backend import clear_session, set_session
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import copy


app = Flask(__name__)
class_dict = ['Choroidal Neovascularization (CNV)', 'Diabetic Macular Edema (DME)', 'DRUSEN', 'NORMAL']
pred_datagen = ImageDataGenerator(rescale=1./255)
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
model = load_model('./model/model.hdf5')
# model._make_predict_function()
# print(model.predict(np.ones((1,256,256,1))))

def predict(img):
    return model.predict(img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def get_result():    
    global model
    global graph
    global sess
    global class_dict
    imageByteString = request.files["picture"].read()
    nparr = np.fromstring(imageByteString, np.uint8)   
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) 
    img = preprocess_image(image)
    predict_img = img.copy()
    predict_img = np.expand_dims(predict_img, axis=2)
    predict_img = np.expand_dims(predict_img, axis=0)
    # print(predict_img.shape)
    # print(predict_img)
    with graph.as_default():
        set_session(sess)
        y_pred = predict(predict_img/255.0)
        # print(y_pred)
    Y_pred = np.argmax(y_pred)
    # print(Y_pred)
    # print(class_dict[Y_pred])
    retval, buffer = cv2.imencode('.png', img)
    # jpg_as_text = base64.b64encode(buffer)
    base64_frame = base64.b64encode(buffer)
    base64_string = base64_frame.decode('utf-8')
    raw_respond = {'image': str(base64_string), 'result': 'Kết quả: ' + class_dict[Y_pred]}
    

    return json.dumps(raw_respond)



def preprocess_image(img, diff=10):
    kernel = np.ones((3,3))
    print(img.shape)
    h, w = img.shape
    h, w = img.shape
        # diff = 10
    for i in range(w):
        if img[0, i] > 220:
            cv2.floodFill(img, None, (i, 0), 0, loDiff=diff, upDiff=diff)

    for i in range(h):
        if img[i, w - 1] > 220:
            cv2.floodFill(img, None, (w-1, i), 0, loDiff=diff, upDiff=diff)

    for i in range(w-1, -1, -1):
        if img[h - 1, i] > 220:
            cv2.floodFill(img, None, (i, h-1), 0, loDiff=diff, upDiff=diff)

    img_org = copy.copy(img)        
    _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = 255 - img
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.dilate(img,kernel,iterations = 7)        
    x1, y1, x2, y2 = findROICoord(img)
    x1 = np.maximum(x1 - 60, 0)
    y1 = np.maximum(y1 - 60, 0)
    x2 = np.minimum(x2 + 60, w - 1)
    y2 = np.minimum(y2 + 60, h - 1)
    return resize(img_org[y1:y2, x1:x2], (256, 256))

def findROICoord(img):
    h, w = img.shape
    x1 = 10000
    y1 = 10000
    x2 = 0
    y2 = 0
    for i in range(h):
        for j in range(w):
            if img[i, j] < 10:
                if i < y1:
                    y1 = i
                if j < x1:
                    x1 = j
                if i > y2:
                    y2 = i
                if j > x2:
                    x2 = j
    return x1, y1, x2, y2

def resize(img, require_size = (128, 128)):
    try:
        if img.shape < require_size:
            resized_img = cv2.resize(img, require_size, cv2.INTER_AREA)
        else:
            resized_img = cv2.resize(img, require_size, cv2.INTER_CUBIC)
    except Exception as e:
        print(str(e))
    return resized_img