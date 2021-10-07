from flask import Blueprint, render_template, request, flash, jsonify, request, Response
from flask_login import login_required, current_user
from .models import Note, Img
from . import db, text_to_speech
import json
from werkzeug.utils import secure_filename
from flask_cors import cross_origin
import cv2
from keras.models import load_model
import numpy as np
import time


views = Blueprint('views', __name__)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE,  5)

model = load_model("C:/Users/HP/Desktop/FLASK DEMO/website/saved_model_mobileNet.h5") # C:/Users/HP/Desktop/FLASK DEMO/website/saved_model_mobileNet.h5
input_size = (224, 224)

def generate_frames():
    while True:
        start_time = time.time()

        success, frame = camera.read()
        if not success:
            break

        model_frame = cv2.resize(frame, input_size, frame)
        model_frame = np.expand_dims(model_frame, axis=0) / 255.0

        classes = model.predict(model_frame)[0]
        label = np.argmax(classes)

        inference_time = time.time() - start_time
        fps = 1 / inference_time

        fps_msg = "Time: {:05.1f}ms {:.1f} FPS".format(inference_time*1000, fps)

        answer = "Answer is "
        if label == 0:
            answer += "1$"
        elif label == 1:
            answer += "10$"
        elif label == 2:
            answer += "50$"
        else:
            answer += "No Detect"

        cv2.putText(frame, fps_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
        cv2.putText(frame, answer, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)


        label = 3
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if(request.method == 'POST'):
        note = request.form.get('note')

        if(len(note) < 1):
            flash('Note is too short!', category='error')
        else:
            new_note = Note(data=note, user_id=current_user.id)
            db.session.add(new_note)
            db.session.commit()
            flash('Note added successfuly!', category='success')

    return render_template("home.html", user=current_user)

@views.route('/upload', methods=['GET', 'POST'])
def uploadImg():
    if(request.method == 'POST'):
        pic = request.files['pic']
        if not pic:
            flash('No picture uploaded.', category='error')
        else:
            filename = secure_filename(pic.filename)
            mimetype = pic.mimetype
            print(filename)
            print(mimetype)
            new_img = Img(img=pic.read(), mimetype=mimetype, name=filename)
            db.session.add(new_img)
            db.session.commit()
            flash('Image has been uploaded successfuly!', category='success')

    return render_template("image.html", user=current_user)

@views.route('/<int:id>')
def get_image(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'No img with that id', 404
    else:
        return Response(img.img, mimetype=img.mimetype)

# text_example = "지금 스캔하신 지폐의 금액을 알려드리겠습니다. 총 1,371,513,000원 입니다."
@views.route('/tts', methods=['GET', 'POST'])
@cross_origin()
def tts():
    if(request.method == 'POST'):
        text = request.form['speech']
        text_to_speech(text)

    return render_template("tts.html", user=current_user)

@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data) # python-dictionary
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if(note):
        if(note.user_id == current_user.id):
            db.session.delete(note)
            db.session.commit()
    return jsonify({})

@views.route('/video-stream')
def vindex():
    return render_template('video.html', user=current_user)

@views.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

