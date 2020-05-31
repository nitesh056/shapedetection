import cv2
import numpy as np
import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.shape = "No shape detected"
        self.getVideoInfo()
        self.hsvRangeTuple = (0,0,0,255,255,255)
        
        self.img_size = 60
        self.pad = 60
        self.model = load_model('D:/shapedetection/data/shapesmodel.h5')
        self.dimData = np.prod([self.img_size, self.img_size])

    def __del__(self):
        self.video.release()

    def largest_contour(self, contours):
        return max(contours, key=cv2.contourArea)[1]

    def contour_center(self, c):
        M = cv2.moments(c)
        try:
            center = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        except:
            center = 0, 0
        return center

    def only_color(self, img, hsvRangeTuple):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower, upper = np.array(hsvRangeTuple[0:3]), np.array(hsvRangeTuple[3:6])
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return res, mask

    def bbox(self, img, c):
        x, y, w, h = cv2.boundingRect(c)
        return img[y-self.pad:y+h+self.pad, x-self.pad:w+x+self.pad], (x, y)

    def get_hsv_range(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        min_h = min_s = min_v = max_h = max_s = max_v = 0
        for point in self.all_points:
            px = frame[point]
            px_array = np.uint8([[px]])
            px_hsv = cv2.cvtColor(px_array, cv2.COLOR_BGR2HSV)
            if (min_h > px_hsv[0][0][0]) | (min_h == 0):
                min_h = px_hsv[0][0][0]
            if (min_s > px_hsv[0][0][1]) | (min_s == 0):
                min_s = px_hsv[0][0][1]
            if (min_v > px_hsv[0][0][2]) | (min_v == 0):
                min_v = px_hsv[0][0][2]
            if (max_h < px_hsv[0][0][0]) | (max_h == 0):
                max_h = px_hsv[0][0][0]
            if (max_s < px_hsv[0][0][1]) | (max_s == 0):
                max_s = px_hsv[0][0][1]
            if (max_v < px_hsv[0][0][2]) | (max_v == 0):
                max_v = px_hsv[0][0][2]

        min_h = (min_h-20 if min_h-20 > 0 else 0)
        min_s = (min_s-20 if min_s-20 > 0 else 0)
        min_v = (min_v-20 if min_v-20 > 0 else 0)
        max_h = (max_h+20 if max_h+20 < 180 else 180)
        max_s = (max_s+20 if max_s+20 < 255 else 255)
        max_v = (max_v+20 if max_v+20 < 255 else 255)

        self.hsvRangeTuple = (min_h, min_s, min_v, max_h, max_s, max_v)

    def get_frame(self):
        _, img = self.video.read()
        imgc = img.copy()
        height, width, _ = img.shape
        self.get_hsv_range(img)
        _, mask = self.only_color(img, self.hsvRangeTuple)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 3000 and area < 1180000:
                roi, coords = self.bbox(img, c)
                if np.prod(roi.shape[:2]) > 10:
                    roi = cv2.resize(roi, (self.img_size, self.img_size))
                    _, roi = self.only_color(roi, self.hsvRangeTuple)
                    roi = 255-roi
                    mask = cv2.resize(roi, (self.img_size, self.img_size))
                    mask = mask.reshape(self.dimData)
                    mask = mask.astype('float32')
                    mask /= 255
                    prediction = self.model.predict(mask.reshape(1, self.dimData))[0].tolist()
                    text = ''
                    p_val, th = .25, .5
                    if max(prediction) > p_val:
                        if prediction[0] > p_val and prediction[0] == max(prediction):
                            text, th = 'triangle', prediction[0]
                        if prediction[1] > p_val and prediction[1] == max(prediction):
                            text, th = 'circle', prediction[1]
                        if prediction[2] > p_val and prediction[2] == max(prediction):
                            text, th = 'rectangle', prediction[2]
                        if prediction[3] > p_val and prediction[3] == max(prediction):
                            text, th = 'circle', prediction[3]
                    cv2.drawContours(imgc, c, -1, (0, 0, 255), 1)
                    org, font, color = (coords[0], coords[1]+int(area/400)), cv2.FONT_HERSHEY_SIMPLEX, (0, 0, 255)
                    cv2.putText(imgc, text, org, font, int(2.2*area/15000), color, int(6*th), cv2.LINE_AA)
                    if text != '':
                        imgc[imgc.shape[0]-200:imgc.shape[0], img.shape[1]-200:img.shape[1]] = cv2.cvtColor(cv2.resize(roi, (200, 200)), cv2.COLOR_GRAY2BGR)
        ret, jpeg = cv2.imencode('.jpg', imgc)
        return jpeg.tobytes()

    def getVideoInfo(self):
        _, frame = self.video.read()
        self.height = frame.shape[0]
        self.width = frame.shape[1]

        self.half_height = int(self.height / 2)
        self.half_width = int(self.width / 2)
        
        self.five_percent_height = int(self.height * 0.025)
        self.five_percent_width = int(self.width * 0.025)
        
        self.center_point = self.half_height, self.half_width
        self.upper_left_point = self.half_height - self.five_percent_height, self.half_width - self.five_percent_width
        self.upper_right_point = self.half_height - self.five_percent_height, self.half_width + self.five_percent_width
        self.lower_left_point = self.half_height + self.five_percent_height, self.half_width - self.five_percent_width
        self.lower_right_point = self.half_height + self.five_percent_height, self.half_width + self.five_percent_width

        self.all_points = [self.center_point,self.upper_left_point, self.upper_right_point, self.lower_left_point, self.lower_right_point]

    def getShape(self):
        return self.shape
