from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

img_size = 60
hsvRangeTuple = (165, 47, 113, 180, 171, 225)
pad = 60
model = load_model('D:/shapedetection/data/shapesmodel.h5')
dimData = np.prod([img_size, img_size])


def largest_contour(contours):
    return max(contours, key=cv2.contourArea)[1]


def contour_center(c):
    M = cv2.moments(c)
    try:
        center = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    except:
        center = 0, 0
    return center

def only_color(img, hsvRangeTuple):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array(hsvRangeTuple[0:3]), np.array(hsvRangeTuple[3:6])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return res, mask

def bbox(img, c):
    x, y, w, h = cv2.boundingRect(c)
    return img[y-pad:y+h+pad, x-pad:w+x+pad], (x, y)

def get_hsv_range(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_h = min_s = min_v = max_h = max_s = max_v = 0
    for point in all_points:
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

    return min_h, min_s, min_v, max_h, max_s, max_v


def getVideoInfo(self):
    _, frame = self.video.read()
    self.height = frame.shape[0]
    self.width = frame.shape[1]

    self.half_height = int(self.height / 2)
    self.half_width = int(self.width / 2)

    self.five_percent_height = int(self.height * 0.025)
    self.five_percent_width = int(self.width * 0.025)

    self.center_point = self.half_height, self.half_width
    self.upper_left_point = self.half_height - \
        self.five_percent_height, self.half_width - self.five_percent_width
    self.upper_right_point = self.half_height - \
        self.five_percent_height, self.half_width + self.five_percent_width
    self.lower_left_point = self.half_height + \
        self.five_percent_height, self.half_width - self.five_percent_width
    self.lower_right_point = self.half_height + \
        self.five_percent_height, self.half_width + self.five_percent_width

    self.all_points = [self.center_point, self.upper_left_point, self.upper_right_point, self.lower_left_point, self.lower_right_point]



def run_frame(img):
    imgc = img.copy()
    height, width, _ = img.shape

    hsvRangeTuple = get_hsv_range(img)
    #mask of the green regions in the image
    _, mask = only_color(img, hsvRangeTuple)

    #find the contours in the image
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #iterate through the contours, "Keras, what shape is this contour?"
    for c in contours:
        #if the contour is too big or too small, it can be ignored
        area = cv2.contourArea(c)
        #print area
        if area > 3000 and area < 1180000:

            #crop out the green shape
            roi, coords = bbox(img, c)

            #filter out contours that are long and stringy
            if np.prod(roi.shape[:2]) > 10:

                #get the black and white image of the shape
                roi = cv2.resize(roi, (img_size, img_size))
                _, roi = only_color(roi, hsvRangeTuple)
                roi = 255-roi  # Keras likes things black on white
                mask = cv2.resize(roi, (img_size, img_size))
                mask = mask.reshape(dimData)
                mask = mask.astype('float32')
                mask /= 255
                #feed image into model
                prediction = model.predict(mask.reshape(1, dimData))[0].tolist()

                #create text --> go from categorical labels to the word for the shape.
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

                #draw the contour
                cv2.drawContours(imgc, c, -1, (0, 0, 255), 1)

                #draw the text
                org, font, color = (
                    coords[0], coords[1]+int(area/400)), cv2.FONT_HERSHEY_SIMPLEX, (0, 0, 255)
                cv2.putText(imgc, text, org, font, int(
                    2.2*area/15000), color, int(6*th), cv2.LINE_AA)

                #paste the black and white image onto the source image (picture in picture)
                if text != '':
                    imgc[imgc.shape[0]-200:imgc.shape[0], img.shape[1]-200:img.shape[1]
                         ] = cv2.cvtColor(cv2.resize(roi, (200, 200)), cv2.COLOR_GRAY2BGR)

    ret, jpeg = cv2.imencode('.jpg', imgc)
    return jpeg.tobytes()
