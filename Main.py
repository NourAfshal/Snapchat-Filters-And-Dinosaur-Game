from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import cv2
import threading
import os
import time
from threading import Thread
from os import listdir
from os.path import isfile, join
import dlib
from imutils import face_utils, rotate_bound
import math


print("Current Working Directory:", os.getcwd())
print("Files in ./sprites/:", os.listdir("./sprites/"))


def put_sprite(num):
    global SPRITES, BTNS
    SPRITES[num] = 1 - SPRITES[num]
    if SPRITES[num]:
        BTNS[num].config(relief=SUNKEN)
    else:
        BTNS[num].config(relief=RAISED)


def draw_sprite(frame, sprite, x_offset, y_offset):
    (h, w) = (sprite.shape[0], sprite.shape[1])
    (imgH, imgW) = (frame.shape[0], frame.shape[1])

    if y_offset + h >= imgH:
        sprite = sprite[0:imgH - y_offset, :, :]

    if x_offset + w >= imgW:
        sprite = sprite[:, 0:imgW - x_offset, :]

    if x_offset < 0:
        sprite = sprite[:, abs(x_offset)::, :]
        w = sprite.shape[1]
        x_offset = 0

    for c in range(3):
        frame[y_offset:y_offset + h, x_offset:x_offset + w, c] = \
            sprite[:, :, c] * (sprite[:, :, 3] / 255.0) + frame[y_offset:y_offset + h, x_offset:x_offset + w, c] * (
                    1.0 - sprite[:, :, 3] / 255.0)
    return frame


def adjust_sprite2head(sprite, head_width, head_ypos, ontop=True):
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])
    factor = 1.0 * head_width / w_sprite
    sprite = cv2.resize(sprite, (0, 0), fx=factor, fy=factor)
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])

    y_orig = head_ypos - h_sprite if ontop else head_ypos
    if y_orig < 0:
        sprite = sprite[abs(y_orig)::, :, :]
        y_orig = 0
    return sprite, y_orig


def apply_sprite(image, path2sprite, w, x, y, angle, ontop=True):
    try:
        sprite = cv2.imread(path2sprite, -1)
        if sprite is None:
            raise ValueError(f"Error loading sprite from path: {path2sprite}")
        sprite = rotate_bound(sprite, angle)
        (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)

        # Experiment with adjusting the coordinates
        x_offset = x  # You may need to adjust this
        y_offset = y_final  # You may need to adjust this

        image = draw_sprite(image, sprite, x_offset, y_offset)
    except Exception as e:
        print(f"Error applying sprite {path2sprite}: {e}")
    return image




def calculate_inclination(point1, point2):
    x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180 / math.pi * math.atan((float(y2 - y1)) / (x2 - x1))
    return incl


def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:, 0])
    y = min(list_coordinates[:, 1])
    w = max(list_coordinates[:, 0]) - x
    h = max(list_coordinates[:, 1]) - y
    return x, y, w, h


def get_face_boundbox(points, face_part):
    if face_part == 1:
        (x, y, w, h) = calculate_boundbox(points[17:22])  # left eyebrow
    elif face_part == 2:
        (x, y, w, h) = calculate_boundbox(points[22:27])  # right eyebrow
    elif face_part == 3:
        (x, y, w, h) = calculate_boundbox(points[36:42])  # left eye
    elif face_part == 4:
        (x, y, w, h) = calculate_boundbox(points[42:48])  # right eye
    elif face_part == 5:
        (x, y, w, h) = calculate_boundbox(points[29:36])  # nose
    elif face_part == 6:
        (x, y, w, h) = calculate_boundbox(points[48:68])  # mouth
    return x, y, w, h


def calculate_iou(pre_face, cur_face):
    (x1, y1, width1, height1) = (pre_face.left(), pre_face.top(), pre_face.width(), pre_face.height())
    (x2, y2, width2, height2) = (cur_face.left(), cur_face.top(), cur_face.width(), cur_face.height())
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)
    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        area = width * height
        area1 = width1 * height1
        area2 = width2 * height2
        ratio = area * 1. / (area1 + area2 - area)
    return ratio


iou_thres = 0.93


def cvloop(run_event):
    global panelA
    global SPRITES
    global image

    dir_ = "./sprites/flyes/"
    flies = [f for f in listdir(dir_) if isfile(join(dir_, f))]
    i = 0
    video_capture = cv2.VideoCapture(0)
    (x, y, w, h) = (0, 0, 10, 10)

    # Filters path
    detector = dlib.get_frontal_face_detector()

    # Facial landmarks
    print("[INFO] loading facial landmark predictor...")
    model = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model)

    pre_face = 0

    while run_event.is_set():
        ret, image = video_capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        if len(faces) > 0:
            cur_face = faces[0]
            if pre_face != 0:
                c_iou = calculate_iou(pre_face, cur_face)
                if c_iou > iou_thres:
                    cur_face = pre_face
            pre_face = cur_face
            face = cur_face
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            incl = calculate_inclination(shape[17], shape[26])

            is_mouth_open = (shape[66][1] - shape[62][1]) >= 20

            if SPRITES[0]:
                apply_sprite(image, "./sprites/hat.png", w, x, y, incl)

            if SPRITES[1]:
                (x1, y1, w1, h1) = get_face_boundbox(shape, 6)
                apply_sprite(image, "./sprites/mustache.png", w1, x1, y1, incl)

            # glasses condition
            if SPRITES[3]:
                (x3, y3, _, h3) = get_face_boundbox(shape, 1)
                apply_sprite(image, "./sprites/glasses.png", w, x, y3, incl, ontop=False)

            # flies condition
            if SPRITES[2]:
                # to make the "animation" we read each time a different image of that folder
                # the images are placed in the correct order to give the animation impression
                apply_sprite(image, dir_ + flies[i], w, x, y, incl)
                i += 1
                i = 0 if i >= len(flies) else i  # when done with all images of that folder, begin again

            # doggy condition
            (x0, y0, w0, h0) = get_face_boundbox(shape, 6)  # bound box of mouth
            if SPRITES[4]:
                (x3, y3, w3, h3) = get_face_boundbox(shape, 5)  # nose
                apply_sprite(image, "./sprites/doggy_nose.png", w3, x3, y3, incl, ontop=False)

                apply_sprite(image, "./sprites/doggy_ears.png", w, x, y, incl)

                if is_mouth_open:
                    apply_sprite(image, "./sprites/doggy_tongue.png", w0, x0, y0, incl, ontop=False)
            else:
                if is_mouth_open:
                    apply_sprite(image, "./sprites/rainbow.png", w0, x0, y0, incl, ontop=False)

        # OpenCV represents an image as BGR; PIL is RGB, we need to change the channel order
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # converts to PIL format
        image = Image.fromarray(image)
        # Converts to a TK format to visualize it in the GUI
        image = ImageTk.PhotoImage(image)
        # Update the image in the panel to show it
        panelA.configure(image=image)
        panelA.image = image

    video_capture.release()


def show_captured_image(img_with_sprites):
    captured_window = Toplevel(root)
    captured_window.title("Captured Image with Filters")

    img_with_sprites = img_with_sprites.astype(np.uint8)

    img_with_sprites_rgb = cv2.cvtColor(img_with_sprites, cv2.COLOR_BGR2RGB)

    img_with_sprites_pil = Image.fromarray(img_with_sprites_rgb)

    img_with_sprites_tk = ImageTk.PhotoImage(img_with_sprites_pil)

    captured_image_label = Label(captured_window, image=img_with_sprites_tk)
    captured_image_label.image = img_with_sprites_tk
    captured_image_label.pack()

    close_button = Button(captured_window, text="Close", command=captured_window.destroy)
    close_button.pack()


image_lock = threading.Lock()


def take_picture():
    global SPRITES
    global image
    global image_lock

    filename = f"Image_{time.strftime('%Y%m%d%H%M%S')}.png"
    model = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model)

    # Acquire the lock before reading the current image
    with image_lock:
        img_with_sprites = np.copy(image)

    if img_with_sprites is not None and img_with_sprites.size != 0:
        (x, y, w, h) = (0, 0, 10, 10)
        gray = cv2.cvtColor(img_with_sprites, cv2.COLOR_BGR2GRAY)

        if len(gray.shape) == 2:  # Ensure that the image is not empty and has a valid shape
            detector = dlib.get_frontal_face_detector()
            faces = detector(gray, 0)

            if len(faces) > 0:
                face = faces[0]
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                incl = calculate_inclination(shape[17], shape[26])

                # Apply mustache sprite
                if SPRITES[1]:
                    (x1, y1, w1, h1) = get_face_boundbox(shape, 6)
                    img_with_sprites = apply_sprite(img_with_sprites, "./sprites/mustache.png", w1, x1, y1, incl)

                # Apply glasses sprite
                if SPRITES[3]:
                    (x3, y3, _, h3) = get_face_boundbox(shape, 1)
                    img_with_sprites = apply_sprite(img_with_sprites, "./sprites/glasses.png", w, x, y3, incl, ontop=False)

                (x0, y0, w0, h0) = get_face_boundbox(shape, 6)  # bound box of mouth
                (x3, y3, w3, h3) = get_face_boundbox(shape, 5)  # nose
                # Apply doggy filter
                if SPRITES[4]:

                    img_with_sprites = apply_sprite(img_with_sprites, "./sprites/doggy_nose.png", w3, x3, y3, incl, ontop=False)
                    img_with_sprites = apply_sprite(img_with_sprites, "./sprites/doggy_ears.png", w, x, y, incl)
                    is_mouth_open = (shape[66][1] - shape[62][1]) >= 20
                    if is_mouth_open:
                        img_with_sprites = apply_sprite(img_with_sprites, "./sprites/doggy_tongue.png", w0, x0, y0, incl, ontop=False)
                else:
                    # Apply rainbow when doggy filter is not selected and mouth is open
                    is_mouth_open = (shape[66][1] - shape[62][1]) >= 20
                    if is_mouth_open:
                        img_with_sprites = apply_sprite(img_with_sprites, "./sprites/rainbow.png", w0, x0, y0, incl, ontop=False)

                for i, sprite_active in enumerate(SPRITES):
                    if sprite_active and i not in [1, 3, 4]:  # Skip mustache (index 1), glasses (index 3), and doggy (index 4)
                        sprite_path = f"./images/sprite_{i + 1}.png"
                        try:
                            # Acquire the lock before modifying the image
                            with image_lock:
                                img_with_sprites = apply_sprite(img_with_sprites, sprite_path, w, x, y, 0)
                        except Exception as e:
                            print(f"Error applying sprite {i}: {e}")

                img_with_sprites = img_with_sprites.astype(np.uint8)

                cv2.imwrite(filename, img_with_sprites)
                print(f"Picture saved as {filename}")

                show_captured_image(img_with_sprites)
            else:
                print("Error: No face detected.")
        else:
            print("Error: Invalid image shape.")
    else:
        print("Error: Empty or None image.")



root = Tk()
root.title("Snapchat filters")
this_dir = os.path.dirname(os.path.realpath(__file__))

btn1 = Button(root, text="Hat", command=lambda: put_sprite(0))
btn1.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn2 = Button(root, text="Mustache", command=lambda: put_sprite(1))
btn2.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn3 = Button(root, text="Flies", command=lambda: put_sprite(2))
btn3.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn4 = Button(root, text="Glasses", command=lambda: put_sprite(3))
btn4.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn5 = Button(root, text="Doggy", command=lambda: put_sprite(4))
btn5.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn_snapshot = Button(root, text="Take Picture", command=take_picture)
btn_snapshot.pack(side="top", fill="both", expand="no", padx="5", pady="5")

panelA = Label(root)
panelA.pack(padx=10, pady=10)

SPRITES = [0, 0, 0, 0, 0]
BTNS = [btn1, btn2, btn3, btn4, btn5]

run_event = threading.Event()
run_event.set()
action = Thread(target=cvloop, args=(run_event,))
action.setDaemon(True)
action.start()


def terminate():
    global root, run_event, action
    print("Closing thread opencv...")
    run_event.clear()
    time.sleep(1)
    root.destroy()
    print("All closed! Chao")


root.protocol("WM_DELETE_WINDOW", terminate)
root.mainloop()
