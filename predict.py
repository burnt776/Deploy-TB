from flask import Flask, render_template, request,redirect
import os,cv2
from keras.models import Model,load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
size=96

def processimg(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size)) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(img)
    crop=cv2.resize(equalized_img,(size,size))
    return crop

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/upload", methods=['POST'])
def upload():
    model=load_model('Model.h5')   
    print("model_loaded")
    target = os.path.join(APP_ROOT, 'static/xray/')
    if not os.path.isdir(target):
        os.mkdir(target)
    filename = ""
    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)
    img = cv2.imread(destination)
    cv2.imwrite('static/xray/file.png',img)
    img= processimg(img)
    cv2.imwrite('static/xray/processedfile.png',img)
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img_to_array(img)
    img = cv2.resize(img,(size,size))
    img=img.reshape(1,size,size,3)
    img = img.astype('float32')
    img = img / 255.0
    result = model.predict(img)
    pred=model.predict(img)
    
    neg=pred[0][0]
    pos=pred[0][1]
    
    classes=['Negative','Positive']

    if(neg>pos):
        return render_template("result.html", pred=classes[0],pos=pos,neg=neg, filename=filename)
        plot_dest = "/".join([target, "result.png"])
    else:
        return render_template("result.html", pred=classes[1],pos=pos,neg=neg, filename=filename)
        plot_dest = "/".join([target, "result.png"])

if __name__ == '__main__':
    app.run(debug=True)
