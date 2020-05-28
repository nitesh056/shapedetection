import cv2
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.shape = "No shape detected"
        self.getVideoInfo()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, frame = self.video.read()
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
            
        lower = np.array([min_h, min_s, min_v])
        upper = np.array([max_h, max_s, max_v])

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            if area > 1000:
                cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                if len(approx) == 3:
                    self.shape = "Triangle"
                elif len(approx) == 4:
                    self.shape = "Rectangle"
                elif len(approx) < 20:
                    self.shape = "Circle"

        cv2.rectangle(frame,
        (
            self.half_width - self.five_percent_width + 2,
            self.half_height - self.five_percent_height + 2
        ),(
            self.half_width + self.five_percent_width + 2,
            self.half_height + self.five_percent_height + 2 
        ),(0, 255, 0), 2)

        cv2.putText(frame, self.shape, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
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