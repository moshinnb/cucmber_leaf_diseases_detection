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
from flask import Flask,request,flash, redirect, url_for, request,jsonify, render_template
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
def home():
    # Main page
    return render_template('index.html')






global preds
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
        
        return preds

    

        
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
    r="No Remedies Present for the given Disease"
    g="No Remedies"
    a="Cause: Fungus (Alternaria cucumerina).  Symptoms: Irregular brown spots on the leaves—sometimes with yellow edges.  Treatment: Fungicides—either commercial or homemade.  You can also try spraying homemade fungicides made of a mixture of soapy water, baking soda, and vinegar. Some organic gardeners also opt for copper fungicides.  If only a few parts of the plant are affected, cut and remove those parts to prevent the fungus from spreading.  If the plant is severely infected, you may have to remove the whole plant, treat or replace the soil, and start over."
    m="Cause: Virus(Cucumber mosaic virus).  Symptoms: White, yellow, or green lines and patterns on leaves and fruit.Plants may not produce as much fruit, and if they do, the fruits may be small and malformed. The leaves may also be malformed. The veins of the leaves may also become very distinct and yellow.    Treatment: There are currently no treatments that can cure or prevent a cucumber mosaic virus infection.Remove any infected plants and plant materials.Remove weeds and plant debris often since both can be breeding grounds for many plant diseases."
    c="No Remedies present "
    w="Cause: Bacteria (Erwinia tracheiphila) & is spread mainly through cucumber beetles (striped and spotted).The bacteria survive winters by living in their beetles' guts.  Symptoms: Yellow, wilting leaves that appear to be drying out.The presence of a milky sap oozing from the cut signifies bacterial wilt.   Treatment: There is currently no effective treatment for bacterial wilt.If a large part of your plant is affected, it's best to remove the entire plant from your garden. Again, prevention is still the best treatment. Make sure you use sanitary cultural practices and avoid exposing stems, leaves, and fruit to water and soil contact."
    p="Cause: Fungus(Podosphaera xanthii or Erysiphe cichoracearum)The fungi favor warm, wet conditions, which is why it is commonly found in greenhouse-grown plants.The spores don’t need moisture to germinate, and the can spread to other plants by wind, insect, and contaminated water and gardening equipment.   Symptoms: White, powdery spots or layers on the leaves and stems.Fruits can also be affected, although this is rare.   Treatment: Spray chemical or organic fungicides and remove affected plant parts.The usual method of spraying fungicide and removing affected plant parts still applies, but there are many organic treatment options that are just as effective like Milk, Neem oil, Sulphur, Copper fungicides, Garlic, Baking Soda, Vinegar, Potassium bicarbonate."
    d="Cause: Fungus(Pseudoperonospora cubensis).Downy mildew favors shade and moisture. The fungi cannot survive extremely cold winters (like those in the Northeastern U.S.), but in temperate regions, they can overwinter in plant debris.   Symptoms: Light green or yellow spots on the leaves that appear angular.You will also find fuzzy, dark gray spots with a purplish tint (spores) on the underside of the leaves—a tell-tale sign of downy mildew. As the disease progresses, leaves will dry out, become brown, and fall off. However, visible symptoms are not always consistent.  Treatment: Spray fungicides ike Orondis, Ranman, Curzate, Zing!, Zampro and remove infected plant parts & Severe infections may require you to remove most or all of the affected plant to prevent further spreading."
    b="Cause:Bacteria(Pseudomonas syringae, Septoria cucurbitacearum, or Xanthomonas campestris)The bacteria that cause leaf spots are opportunistic, spreading through lesions created by insect bites and through seeds that spread the bacteria through contact with water. However, they are not as hardy as other pathogens.    Symptoms: Small, brownish, angular or circular spots—sometimes with yellow edges—or black spots on the leaves.Xanthomonas infection usually causes brown spots with yellow edges, while Pseudomonas infection usually causes reddish-brown spots. In both cases, spots will quickly turn black.Spots on the leaves and fruit can give other pathogens—especially fungi—the opportunity to infect. This disease often causes leaves to fall off, making fruits vulnerable to sunscalding.   Treatment:There is currently no effective treatment for bacterial leaf spots.Look for cucumber beetles starting early in the spring, when the weather starts to warm up. They usually come out in early evening. Remove them as you see them."
    n="Cause: Fungus(Colletotrichum orbiculare).Regions with high humidity like southern and mid-Atlantic states are highly susceptible to fungal plant diseases.The spores (conidia) need moisture and mild temperatures to germinate.   Symptoms: Yellow, water-soaked, circular spots on the leaves and fruit with dark brown to black edges.Spots on the fruits may appear black and sunken with a pink- or salmon-colored gelatinous substance in the center. This gelatinous substance is a cluster of fungal spores—a characteristic sign of anthracnose.   Treatment: Fungicides like Chlorothalonil (Bravo) and benomyl (Benlate) are popular fungicides used to treat anthracnose. In rainy seasons, more frequent applications may be required.If a large part of your plant is infected, you may need to remove the entire plant and start over with sanitized soil."
    if preds == "good_cucumber":
        return g
    elif preds == "Altrnaria_blight":
        return a
    elif preds == "cucumber_mosic_virus":
        return m
    elif preds == "Cercospora_LF":
        return c
    elif preds == "bacterial_wilt":
        return w
    elif preds == "powdery_mildew":
        return p
    elif preds == "downy_mildew":
        return d
    elif preds == "bacteral_leaf_spot":
        return b
    elif preds == "anthracnose":
        return n
    else:
        return r

if __name__ == "__main__":
    app.run()
    
