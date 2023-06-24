from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import ast
import time

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


def save_shape_to_lot(lot_name, shape):
    path = f"/Users/leonardomosimannconti/computer_vision/parking_spot_detection/spots/{lot_name}.txt"
    with open(path, "a+") as f:
        f.write(str(shape) + " 0")
        f.write("\n")

    return shape


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
        print(e)
        return [], []


def draw_shapes(image, shapes, colors):
    for i, shape in enumerate(shapes):
        cv2.polylines(image, [shape], isClosed=True,
                      color=colors[i], thickness=2)
        cv2.putText(image, f"{i}", (shape[0][0][0], shape[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return image


def draw_spots(image, spots):
    for i, spot in enumerate(spots):
        cv2.rectangle(image, (int(spot[0][0]), int(spot[0][1])), (int(spot[1][0]), int(spot[1][1])), (255, 255, 0), 2)
        cv2.putText(image, f"{i}", (int(spot[0][0]), int(spot[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return image

def draw_drawing_lines(image, drawing_lines):
    # draw the starting point too
    if last_point is not None:
        cv2.line(image, last_point, curr_point, (0, 255, 0), 2)

    for line in drawing_lines:
        cv2.line(image, line[0], line[1], (0, 255, 0), 2)


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


def change_occupied(index, occupied):  #  fix here
    path = f"/Users/leonardomosimannconti/computer_vision/parking_spot_detection/spots/{lot_name}.txt"
    lines = []
    with open(path, "r") as f:
        lines = f.readlines()
    
    line = lines[index].rstrip('\n')

    line = line[:-1] + occupied
    lines[index] = line + '\n'

    with open(path, "w") as f:
        f.writelines(lines)

    return True
        
def check_spots(spots, drawn_boxes):
    # Aqui farei uma funcao para calcular se a bounding box esta dentro de alguma vaga
    # Se estiver em uma vaga, a cor da vaga deve mudar para vermelho

    # Check for the corresponding bounding box for each spot by checking if one coordinate is inside another,
    # But as the values may vary, we'll need a function that compares in a range which spot should be the corresponding one
    result = []
    # If there's a spot inside the bounding box, change the color of the spot to red


    for i_box, box in enumerate(drawn_boxes):
        # As the box is drawn, the points we're going to look are the top left and bottom right, how to get that?
        # print(box.tolist())
        box = box.tolist()
        box = [subitem for sublist in box for subitem in sublist]
        
        largest_area = 0
        index_box = 0
        index_spot = 0
        occupied = False
        for i_spot, spot in enumerate(spots):
            # Check if the object is inside the bounding box based on the x and y points
            # But only checking if all the points are inside the bounding box is not enough, as the bounding box may be bigger than the spot
            # Or the spot may have been detected outside the box
            # So we'll need to check which detected spot is the most reasonable to be the corresponding one
            # In order to not use too much processing in the loop, we'll limit the
 
            box_x1, box_x2, box_y1, box_y2 = box[0][0], box[1][0], box[0][1], box[1][1]  # 
            spot_x1, spot_x2, spot_y1, spot_y2 = spot[0][0], spot[1][0], spot[0][1], spot[1][1]  # should be good to go
            
            # check if the distance between the box and the spot is greater than 3 times the spot length
            if box_x1 > 3*spot_x1 or box_x1 < spot_x1/3:
                continue

            if box_x2 > spot_x2:
                # ight = True
                abs_x = abs(spot_x1 - box_x2)
            else:
                # right = False
                abs_x = abs(box_x1 - spot_x2)
            if box_y2 > spot_y2:
                # up = False
                abs_y = abs(spot_y2 - box_y1)
            else:
                # up = True
                abs_y = abs(box_y2 - spot_y1)                

            area = abs_x * abs_y
            
            if area > largest_area:
                largest_area = area
                index_box = i_box
                index_spot = i_spot
                occupied = True
        
        if occupied:
            # spots.remove(spots[index_spot])
            change_occupied(i_box, "1")
            result.append((index_box, index_spot))
        
        else:
            change_occupied(i_box, "0")
            result.append((index_box, -1))
            
    # print(result)
    return result



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
                check_spots(spots, shapes)
                first_run = False
                
            # spots = run_yolo(frame)
            # check_spots(spots, shapes)
            draw_shapes(frame, shapes, colors)
            draw_spots(frame, spots)
            print("SPOTS: ", len(spots))
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
