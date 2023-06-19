from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import ast

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

# https://www.youtube.com/watch?v=U7HRKjlXK-Y
# start the detection and display in real time

# Load Model
# model = YOLO("yolov5nu.pt")
# model = torch.hub.load('ultralytics/yolov8', 'yolov8s')
model = YOLO("yolov8s.pt")


def save_shape_to_lot(lot_name, shape):
    path = f"/Users/leonardomosimannconti/computer_vision/parking_spot_detection/spots/{lot_name}.txt"
    with open(path, "a+") as f:
        print(str(shape))
        f.write(str(shape))
        f.write("\n")

    return shape


def get_shapes_from_lot(lot_name):
    path = f"/Users/leonardomosimannconti/computer_vision/parking_spot_detection/spots/{lot_name}.txt"
    shapes = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()  # line example = ((x, y), (x, y) , (x, y), (x, y))
                shape = ast.literal_eval(line)
                shape = np.array(shape)
                # -1 pega o tamanho valor automaticamente
                shape = shape.reshape((-1, 1, 2))
                shapes.append(shape)
        return shapes

    except Exception as e:
        print(e)
        return []


def draw_shapes(image, shapes):
    for shape in shapes:
        cv2.polylines(image, [shape], isClosed=True,
                      color=(0, 255, 0), thickness=2)
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

        # if there are 4 points, a shape was drawn
        if len(points) == 4:
            points_of_shape = np.array(points)
            # -1 pega o tamanho valor automaticamente
            points_of_shape = points_of_shape.reshape((-1, 1, 2))

            save_shape_to_lot(lot_name, points)
            points = []


def check_spots(spots, bounding_boxes):
    # Aqui farei uma funcao para calcular se a bounding box esta dentro de alguma vaga
    # Se estiver em uma vaga, a cor da vaga deve mudar para vermelho

    pass


def run_yolo(image):
    global center_points, frame
    frame = image
    #  results = model.predict(source=image, save=False, show=True, conf=0.3, classes=[])
    results = model.predict(source=frame, save=False, show=False, conf=0.3)

    boxes = results[0].boxes
    for box in boxes:
        # Aqui pegar o ponto central das boxes e comparar com as bounding boxes presentes
        # Na imagem, que foram desenhadas pelo usuario
        # Se o centro da box estiver dentro de alguma bounding box, entao a vaga esta ocupada
        # Assim mudamos a cor para vermelho.

        box.xyxy

        # center_point = (
        #     int((box.xyxy[0] + box.xyxy[2]) / 2), int((box.xyxy[1] + box.xyxy[3]) / 2))

        # example of box xyxy = tensor([[169.1242, 376.8954, 221.2360, 423.7949]])
        print(box.xyxy)
        # center_points.append(center_point)


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

            shapes = get_shapes_from_lot(lot_name)
            draw_shapes(frame, shapes)
            draw_drawing_lines(frame, drawing_lines)
            cv2.imshow("image", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    else:
        shapes = get_shapes_from_lot(lot_name)
        while True:
            ret, frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            run_yolo(frame)
            draw_shapes(frame, shapes)
            cv2.imshow("image", frame)
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
