from flask import Flask, render_template, Response, jsonify
import cv2
# from flask_sqlalchemy import SQLAlchemy
from camerav2 import *

app = Flask(__name__)

# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'

# db = SQLAlchemy(app)

# @app.route('/get_json', methods=['POST'])
# def gettingJson():
#     vid = VideoCamera()
#     vid.get_frame()
#     shape = vid.getShape()
#     return jsonify({'data': render_template('json.html', data=shape)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def start_detection():
    return render_template('detection.html')

def gen():
    cap=cv2.VideoCapture(0)

    while True:
        try:
            _, img = cap.read()
            run_frame(cap)
            # video_frame = camera.get_frame()
            # yield (b'--frame\r\n'
            #     b'Content-Type: image/jpeg\r\n\r\n' + video_frame + b'\r\n\r\n')
        except:
            print("Exception")
            # camera = VideoCamera()

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
