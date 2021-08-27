import h5py
import os
import glob
import cv2
import mahotas
import numpy as np


from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')
# next file


from flask.helpers import flash
from lodingmodel import *
from flask import Flask
from flask import render_template
import os
from flask import request


# Flask utils
from flask import Flask,request,flash, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


import joblib


#loding model


bins=8
fixed_size             = tuple((500, 500))
labels_name=['Altrnaria_blight', 'Cercospora_LF', 'anthracnose',
       'bacteral_leaf_spot', 'bacterial_wilt', 'cucumber_mosic_virus',
       'downy_mildew', 'good_cucumber', 'powdery_mildew']


le=LabelEncoder()
le.fit_transform(labels_name)


model_path=r"models/randomfores.joblib"




def rgb_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img
def bgr_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img



def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()



#test=r'C:\Users\moshi\webpage\sample_images\download (1).jpg'

def model_result(file_path):

    fixed_size             = tuple((500, 500))
    bins                   = 8

    image1 = cv2.imread(file_path)
    image = cv2.resize(image1 ,(500,500))


    #test_global_feature     = rgb_bgr(image)
    RGB_BGR       = rgb_bgr(image)
    BGR_HSV       = bgr_hsv(RGB_BGR)
    IMG_SEGMENT   = img_segmentation(RGB_BGR,BGR_HSV)


    fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
    fv_haralick   = fd_haralick(IMG_SEGMENT)
    fv_histogram  = fd_histogram(IMG_SEGMENT)
    test_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])





    #loding model 
    predict_test=joblib.load(model_path)
    #testing using the loded model

    res=predict_test.predict(test_feature.reshape(1,-1))

    print(res)
    res= (le.inverse_transform(res)[0])
    return res







#--------------------------------


UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key="secret key"









ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	






@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')







@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['file']
    if file.filename == '':
	    flash('No image selected for uploading')
	    return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filename1 = "static/uploads/" + filename  
        preds = model_result(filename1)
        result=preds
        return result

    

        
        #file_path = os.path.join(
        #    UPLOAD_FOLDER, secure_filename(f.filename))
        #f.save(file_path)

        # Make prediction
        #preds = model_result(file_path)
        #result=preds
        #return result
    
    
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)




@app.route('/predict/remedies')
def show():
    r="mohsin nb"
    return r

if __name__ == "__main__":
    app.run(port=3000,debug=True)
    
