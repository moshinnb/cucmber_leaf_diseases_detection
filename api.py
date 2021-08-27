from lodingmodel import *


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
    app.run(debug=True)
    
