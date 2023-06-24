from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import ast
import time
from shapely.geometry import Polygon

points = []
center_points = []
shapes = []
current_rectangle = None
drawing_lines = []

curr_point = None
last_point = None
qtd_points = 0
curr_line = []
drawing = False

spots_detected = []

# https://www.youtube.com/watch?v=U7HRKjlXK-Y
# start the detection and display in real time

# Load Model
# model = YOLO("yolov5nu.pt")
# model = torch.hub.load('ultralytics/yolov8', 'yolov8s')
model = YOLO("yolov8s.pt")


# OK
def save_shape_to_lot(lot_name, shape):
    path = f"/Users/leonardomosimannconti/computer_vision/parking_spot_detection/spots/{lot_name}.txt"
    with open(path, "a+") as f:
        f.write(str(shape) + " 0")
        f.write("\n")

    return shape


# OK
def get_shapes_from_lot(lot_name):
    path = f"/Users/leonardomosimannconti/computer_vision/parking_spot_detection/spots/{lot_name}.txt"
    shapes = []
    colors = []
    try:
        with open(path, "r") as f:
            for line in f:
                line_f = line.strip()[:-1]  # line example = ((x, y), (x, y) , (x, y), (x, y)) 0
                shape = ast.literal_eval(line_f)
                shape = np.array(shape)
                # -1 pega o tamanho do vetor automaticamente
                shape = shape.reshape((-1, 1, 2))
                shapes.append(shape)
                if line[-2] == "1":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                    
                colors.append(color)
                
        return shapes, colors

    except Exception as e:
        return [], []


# OK
def draw_shapes(image, shapes, colors):
    for i, shape in enumerate(shapes):
        cv2.polylines(image, [shape], isClosed=True,
                      color=colors[i], thickness=2)
        cv2.putText(image, f"{i}", (shape[0][0][0], shape[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return image


# OK
def draw_spots(image, spots):
    for i, spot in enumerate(spots):
        cv2.rectangle(image, (int(spot[0][0]), int(spot[0][1])), (int(spot[1][0]), int(spot[1][1])), (255, 255, 0), 2)
        cv2.putText(image, f"{i}", (int(spot[0][0]), int(spot[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return image

# OK 
def draw_drawing_lines(image, drawing_lines):
    # draw the starting point too
    if last_point is not None:
        cv2.line(image, last_point, curr_point, (0, 255, 0), 2)

    for line in drawing_lines:
        cv2.line(image, line[0], line[1], (0, 255, 0), 2)

# OK
def click_and_crop(event, x, y, flags, param):
    global points, frame, drawing_lines, curr_line, starting_point, curr_point, drawing, last_point
    curr_point = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        clicked_point = (x, y)
        points.append(clicked_point)

        # to draw lines in "real time"
        curr_line.append(clicked_point)
        last_point = clicked_point

        # if a line is drawn (2 points), add it to the lines list
        if len(curr_line) == 2:
            drawing_lines.append(curr_line)

            # Se completou o shape cria uma nova linha, sera a cada 3 pois ele completa a ultima
            if len(drawing_lines) % 3 == 0:
                curr_line = []
                last_point = None

            else:
                curr_line = [last_point]

        # if there are 4 points, means the shape is drawn
        if len(points) == 4:
            points_of_shape = np.array(points)
            # -1 pega o tamanho valor automaticamente
            points_of_shape = points_of_shape.reshape((-1, 1, 2))

            save_shape_to_lot(lot_name, points)
            points = []

# Ok
def change_occupied(index, occupied):  #  fix here
    path = f"/Users/leonardomosimannconti/computer_vision/parking_spot_detection/spots/{lot_name}.txt"
    lines = []
    with open(path, "r") as f:
        lines = f.readlines()
    
    line = lines[index].rstrip('\n')

    line = line[:-1] + str(occupied)
    lines[index] = line + '\n'

    with open(path, "w") as f:
        f.writelines(lines)

    return True

# OK, i think so
def sort_points(polygon):
    polygon.sort(key=lambda point: point[1])
    top = sorted(polygon[:2], key=lambda point: point[0])     # sort by x
    bottom = sorted(polygon[2:], key=lambda point: point[0])  # sort by x
    return [top[0], top[1], bottom[1], bottom[0]] 

# TODO: verify this part:
def get_iou_poly(p1, p2):
    p1 = sort_points(p1)
    p2 = sort_points(p2)
    poly1 = Polygon(p1)
    poly2 = Polygon(p2)

    if not poly1.intersects(poly2): # if they don't intersect, return 0
        return 0

    iou = poly1.intersection(poly2).area / poly1.union(poly2).area
    return iou

# TODO: verify this part
def check_spots(spots, drawn_boxes):
    checked_boxes = []
    for spot in spots:
        spot_polygon = [(int(spot[i][0]), int(spot[i][1])) for i in range(4)] # assuming spot has 4 points in clockwise or anticlockwise order
        # print(spot_polygon)
        largest_iou = 0
        index_box = -1
        
        for i_box, box in enumerate(drawn_boxes):
            box = box.tolist()
            box_polygon = [(subitem[0], subitem[1]) for sublist in box for subitem in sublist]

            iou = get_iou_poly(box_polygon, spot_polygon)
            if iou > largest_iou and box not in checked_boxes:
                largest_iou = iou
                index_box = i_box

        if index_box >= 0: # threshold can be adjusted depending on use case
            change_occupied(index_box, 1)
            checked_boxes.append(drawn_boxes[index_box].tolist())
        else:
            change_occupied(index_box, 0)
    return checked_boxes

# OK
def run_yolo(image):
    # print("YOLO RAN AT: ", time.time())
    spots_detected = []
    #  results = model.predict(source=image, save=False, show=True, conf=0.3, classes=[])
    results = model.predict(source=image, save=False, show=False, conf=0.3, classes=[2, 7])

    boxes = results[0].boxes
    for box in boxes:
        box = box.xyxy.numpy()[0]
        spots_detected.append([(box[0], box[1]), (box[2], box[3]),
                               (box[0], box[3]), (box[2], box[1])])

    return spots_detected

# OK
def main(args):
    global lot_name, frame

    filepath = args["filepath"]
    webcam_index = args["webcam"]
    drawing_mode = args["draw"]
    lot_name = args["name"]

    if webcam_index is not None:
        video = cv2.VideoCapture(webcam_index)
    elif filepath is not None:
        video = cv2.VideoCapture(filepath)
    else:
        raise Exception(
            "No video source provided, please enter webcam index or filepath")

    if drawing_mode == 1:
        while True:
            ret, frame = video.read()
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", click_and_crop)
            if not ret:
                # If the video ends, reset the video capture to the beginning
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            shapes, colors = get_shapes_from_lot(lot_name)
            draw_shapes(frame, shapes, colors)
            draw_drawing_lines(frame, drawing_lines)
            cv2.imshow("image", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    else:
        # Add start time and current time, so it doesn't run yolo each frame
        first_run = True
        start_time = time.time()

        while True:
            shapes, colors = get_shapes_from_lot(lot_name)
            ret, frame_orig = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame = frame_orig.copy()

            
            curr_time = time.time()
            time_elapsed = curr_time - start_time
            if time_elapsed >= 1 or first_run:
                start_time = curr_time
                spots = None
                spots = run_yolo(frame_orig)
                checked = check_spots(spots, shapes)
                print(checked)
                first_run = False
                
            draw_shapes(frame, shapes, colors)
            draw_spots(frame, spots)
            cv2.imshow("image", frame)
            frame = None
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

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
        "filepath": "/Users/leonardomosimannconti/computer_vision/parking_spot_detection/pklot_video (1).mp4",
        # "filepath": "/Users/leonardomosimannconti/computer_vision/parking_spot_detection/pklot_video.mp4",
        "draw": 0,
        "name": "pk1",
        "webcam": None
    }

    main(args)
