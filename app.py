from flask import Flask, render_template, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
from camera import *

app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'

db = SQLAlchemy(app)

class TrafficSign(db.Model):
    __tablename__ = 'traffic_sign'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20))
    image_name = db.Column(db.String(50))
    shape = db.Column(db.String(10))

class SignCategorization(db.Model):
    __tablename__ = 'sign_categorization'
    id = db.Column(db.Integer, primary_key=True)
    categorization = db.Column(db.String(30))
    shape = db.Column(db.String(10))
    description = db.Column(db.String(100))

@app.route('/get_json', methods=['GET'])
def gettingJson():
    shape = run_frame(cv2.VideoCapture(0), True)
    if shape != '':
        trafficSigns = TrafficSign.query.filter_by(shape=shape)
        signCategorizations = SignCategorization.query.filter_by(shape=shape)
        return jsonify({'data': render_template('shapeDescription.html', shapeDetected=True, trafficSigns=trafficSigns, signCategorization=signCategorizations)})
    else:
        return jsonify({'data': render_template('shapeDescription.html', shapeDetected=False)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def start_detection():
    return render_template('detection.html')

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        try:        
            video_frame = run_frame(cap)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + video_frame + b'\r\n\r\n')
        except:
            cap.release()
            cap = cv2.VideoCapture(0)

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
