from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import refactoring_image_client as Triton

app = Flask(__name__)

FLASK_FOLDER = 'C:/Users/2019A00298/Desktop/flask_upload_img/'
UPLOAD_FOLDER = 'static/uploads/'
IMG_PATH = ""
MODEL_NAME = "mobilenet_money_detection"
ANSWER = ""

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def upload_image():
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
        IMG_PATH = FLASK_FOLDER + os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('@IMG_PATH: ', IMG_PATH)
        print('@upload_image_filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        triton = Triton.Triton(MODEL_NAME, IMG_PATH); triton()
        ANSWER = triton.final_result[-1]

        print("@Flask: Answer is", ANSWER)
        return render_template('index.html', filename=filename, answer=ANSWER)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    print('@display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)