import cv2
import numpy as np


class VideoCamera(object):
    def __init__(self):
       self.video = cv2.VideoCapture(0)
       self.shape = "No shape detected"

    def __del__(self):
        self.video.release()
        self.video.destoryAllWindows()

    def get_frame(self):
        _, frame = self.video.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x, y = 300, 250
        
        # cv2.rectangle(frame, (x, y), (x, y), (0, 0, 0), 2)
        px = frame[x,y]
        px_array = np.uint8([[px]])
        px_hsv = cv2.cvtColor(px_array, cv2.COLOR_BGR2HSV)

        min_h = (px_hsv[0][0][0]-20 if px_hsv[0][0][0]-20 > 0 else 0)
        min_s = (px_hsv[0][0][1]-20 if px_hsv[0][0][1]-20 > 0 else 0)
        min_v = (px_hsv[0][0][2]-20 if px_hsv[0][0][2]-20 > 0 else 0)
        max_h = (px_hsv[0][0][0]+20 if px_hsv[0][0][0]+20 < 180 else 180)
        max_s = (px_hsv[0][0][1]+20 if px_hsv[0][0][1]+20 < 255 else 255)
        max_v = (px_hsv[0][0][2]+40 if px_hsv[0][0][2]+20 < 255 else 255)

        lower = np.array([min_h, min_s, min_v])
        upper = np.array([max_h, max_s, max_v])

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            if area > 400:
                cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                if len(approx) == 3:
                    self.shape = "Triangle"
                elif 3 < len(approx) >= 10:
                    self.shape = "Rectangle"
                elif 10 < len(approx) < 20:
                    self.shape = "Circle"

        cv2.putText(frame, self.shape, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
        
