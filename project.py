import tkinter as tk
import json
import time
import keyboard
import cv2
import numpy as np
from selenium import webdriver


class TRex():

    def __init__(self, data):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument(f"webdriver.chrome.driver={data['driver_path']}")
        self.driver = webdriver.Chrome(options=chrome_options)

    def open_game(self):
        self.driver.get("https://chromedino.com/")
        self.driver.set_window_size(900, 430)

    def play(self):
        cap = cv2.VideoCapture(0)
        time.sleep(5)

        while True:
            ret, img = cap.read()
            height, width, _ = img.shape
            blur = cv2.blur(img, (3, 3))
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            lower = np.array([0, 48, 80])
            upper = np.array([20, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            kernel_square = np.ones((11, 11), np.uint8)
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # Perform morphological transformations to filter out the background noise
            dilation = cv2.dilate(mask, kernel_ellipse, iterations=1)
            erosion = cv2.erode(dilation, kernel_square, iterations=1)
            dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
            filtered = cv2.medianBlur(dilation2, 5)
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
            kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
            median = cv2.medianBlur(dilation2, 5)
            ret, thresh = cv2.threshold(median, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (122, 122, 0), 3)
            new = contours
            max_area = 100
            ci = 0
            if len(contours) > 0:
                for i in range(len(contours)):
                    cnt = contours[i]
                    area = cv2.contourArea(cnt)
                    if area > max_area:
                        max_area = area
                        ci = i
                # Largest area contour
                cnts = new[ci]
                moments = cv2.moments(cnts)

                # Central mass of first-order moments
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                    cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
                centerMass = (cx, cy)
                print(cx, cy)
                cv2.circle(img, centerMass, 7, [100, 0, 255], 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'Center', tuple(centerMass), font, 2, (0, 0, 0), 2)

                if cy > (height / 2) + 70:
                    keyboard.press_and_release('space')
                elif cy < (height / 2) - 70:
                    keyboard.press_and_release('space')

            cv2.line(img, (0, int((height / 2) - 70)), (width, int((height / 2) - 70)), (255, 255, 255), 5)
            cv2.line(img, (0, int(height / 2) + 70), (width, int(height / 2) + 70), (255, 255, 255), 5)
            cv2.imshow("Winks Found", img)
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def destroy_window(self):
        """Custom method to destroy the Tkinter window"""
        self.driver.quit()
        root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Dinosaur Game Controller")

    with open('config.json') as config_file:
        data = json.load(config_file)

    game = TRex(data)
    game.open_game()
    time.sleep(1)
    game.play()

    root.protocol("WM_DELETE_WINDOW", game.destroy_window)  # Bind window close event to the custom method

    root.mainloop()
