from flask import Response
from flask import Flask
from flask import render_template
import cv2


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
