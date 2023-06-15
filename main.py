import cv2
import numpy as np
import argparse
import os
import ast

rectangles = []
current_rectangle = None
drawing = False
image = None
clone = None


def get_rectangles_from_lot(lot_name):
    path = f"/Users/leonardomosimannconti/computer_vision/parking_spot_detection/spots/{lot_name}.txt"

    rectangles = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                # line example = ((x, y), (x, y))
                rectangle = ast.literal_eval(line)
                rectangles.append(rectangle)

        print("Get rectangles from lot")
        print(rectangles)
        return rectangles
    except Exception as e:
        print(e)
        return []


def save_rectangles_to_lot(lot_name, rectangles):
    path = f"/Users/leonardomosimannconti/computer_vision/parking_spot_detection/spots/{lot_name}.txt"
    with open(path, "a+") as f:
        for rectangle in rectangles:
            print(rectangle)
            f.write(str(rectangle))
            f.write("\n")

    return rectangles


def draw_rectangles(image, rectangles):
    for rect in rectangles:
        cv2.rectangle(image, rect[0], rect[1], (0, 255, 0), 2)
    return image


def click_and_crop(event, x, y, flags, param):
    global drawing, image, clone, rectangles, current_rectangle

    # if the left mouse button is clicked, record the starting points and start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        current_rectangle = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Use the clone because the image will be updating with the rectangles
            image = clone.copy()
            cv2.rectangle(image, current_rectangle[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("image", image)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that the rectangle drawing has finished
        current_rectangle.append((x, y))
        drawing = False  # cropping is finished

        cv2.rectangle(
            image, current_rectangle[0], current_rectangle[1], (0, 255, 0), 2)
        clone = image.copy()
        # Update the clone so when drawing the next rectangle, the previous ones are still there
        # Show the image with the rectangle for no delay so the user can draw the next rectangle
        cv2.imshow("image", image)
        # Add the rectangle to the list of rectangles
        # Do not append very small rectangles
        if abs(current_rectangle[0][0] - current_rectangle[1][0]) > 10 and abs(current_rectangle[0][1] - current_rectangle[1][1]) > 10:
            rectangles.append(tuple(current_rectangle))


def main(args):
    global image, clone, lot_name, rectangles

    filepath = args["filepath"]
    webcam_index = args["webcam"]
    drawing_mode = args["draw"]
    lot_name = args["name"]

    rectangles = get_rectangles_from_lot(lot_name)

    if webcam_index is not None:
        video = cv2.VideoCapture(webcam_index)
    elif filepath is not None:
        video = cv2.VideoCapture(filepath)
    else:
        raise Exception(
            "No video source provided, please enter webcam index or filepath")

    if drawing_mode == 1:
        while True:
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", click_and_crop)
            ret, frame = video.read()
            if not ret:
                # If the video ends, reset the video capture to the beginning
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            clone = frame.copy()
            draw_rectangles(frame, rectangles)
            cv2.imshow("image", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    else:
        # start the detection and display in real time
        pass

    print(rectangles)
    save_rectangles_to_lot(lot_name, rectangles)
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # -f parking_spot_detection/pklot_video.mp4 -d 1 -n pk1

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-f", "--filepath", type=str, default=None,
    #                 help="Where's the recording at")
    # ap.add_argument("-w", "--webcam", type=int, default=None,
    #                 help="Index of the webcam on system")
    # ap.add_argument("-d", "--draw", type=int, default=0,
    #                 help="Draw the rectangles (0: No / 1: Yes)", required=True)
    # ap.add_argument("-n", "--name", type=str, default="",
    #                 help="Name of the parking lot", required=True)

    # args = vars(ap.parse_args())

    args = {
        "filepath": "/Users/leonardomosimannconti/computer_vision/parking_spot_detection/pklot_video.mp4",
        "draw": 1,
        "name": "pk1",
        "webcam": None
    }

    main(args)
