import cv2
import numpy as np

rectangles = []
current_rectangle = []
drawing = False
image = None
clone = None

# Function to draw the rectangles in the image
def click_and_crop(event, x, y, flags, param):
    # instead of leaving as global maybe can create a class or something and return rectangle, or not, just use global
    global rectangles, current_rectangle, drawing, image, clone

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
        drawing = False # cropping is finished

        # draw a rectangle around the region of interest
        if abs(current_rectangle[0][0] - current_rectangle[1][0]) > 10 and abs(current_rectangle[0][1] - current_rectangle[1][1]) > 10:  # avoid very small boxes
            cv2.rectangle(image, current_rectangle[0], current_rectangle[1], (0, 255, 0), 2)
            clone = image.copy()
            # Update the clone so when drawing the next rectangle, the previous ones are still there
            # Show the image with the rectangle for no delay so the user can draw the next rectangle
            cv2.imshow("image", image)
            # Add the rectangle to the list of rectangles
            rectangles.append(tuple(current_rectangle))


# load the image, clone it, and setup the mouse callback function
image = cv2.imread("yolov8/2013-03-06_07_05_01.jpg")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    # display the image and wait for a keypress to do an action or break
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # # if the 'r' key is pressed, reset the cropping region
    # if key == ord("r"):
    #     image = clone.copy()

    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

# # if there are two points, then crop the region of interest
# # from the image and display it
# for rect in rectangles:
#     roi = clone[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
#     cv2.imshow("ROI", roi)
#     cv2.waitKey(0)

cv2.destroyAllWindows()
