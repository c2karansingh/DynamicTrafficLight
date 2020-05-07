import cv2
import numpy as np
from flask import Flask, request, render_template




initialize = True
net = None
classes = None
COLORS = np.random.uniform(0, 255, size=(80, 3))

def populate_class_labels():

    class_file_abs_path = "yolov3_classes.txt"
    f = open(class_file_abs_path, 'r')
    classes = [line.strip() for line in f.readlines()]

    return classes


def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_bbox(img, bbox, labels, confidence, colors=None, write_conf=False):

    global COLORS
    global classes

    if classes is None:
        classes = populate_class_labels()

    for i, label in enumerate(labels):

        if colors is None:
            color = COLORS[classes.index(label)]
        else:
            color = colors[classes.index(label)]

        if write_conf:
            label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'

        cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)

        cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def detect_common_objects(image, confidence=0.5, nms_thresh=0.3, model='yolov3', enable_gpu=False):

    Height, Width = image.shape[:2]
    scale = 0.00392

    global classes

    config_file_abs_path = "yolov3.cfg"
    weights_file_abs_path = "yolov3.weights"
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    global initialize
    global net

    if initialize:
        classes = populate_class_labels()
        net = cv2.dnn.readNet(weights_file_abs_path, config_file_abs_path)
        initialize = False

    # enables opencv dnn module to use CUDA on Nvidia card instead of cpu
    if enable_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            max_conf = scores[class_id]
            if max_conf > confidence:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(max_conf))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)

    bbox = []
    label = []
    conf = []

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        bbox.append([round(x), round(y), round(x+w), round(y+h)])
        label.append(str(classes[class_ids[i]]))
        conf.append(confidences[i])

    return bbox, label, conf



app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':

        image_file = request.files['imagefile']
        filename = image_file.filename
        filepath = 'static/uploads/'+filename
        image_file.save(filepath)

        im = cv2.imread(filepath)
        bbox, label, conf = detect_common_objects(im)
        output_image = draw_bbox(im, bbox, label, conf)
        cv2.imwrite('static/outputs/'+filename,output_image)


        print(label)
        total_vehicles = label.count('car') + label.count('bus') + label.count('truck') + label.count('motorcycle')
        print('Number of cars in the image is '+ str(label.count('car')))
        print('Number of buses in the image is '+ str(label.count('bus')))
        print('Number of trucks in the image is '+ str(label.count('truck')))
        print('Number of motorcycles in the image is '+ str(label.count('motorcycle')))
        print("total vehicles",total_vehicles)
        max_seconds = 30
        max_vehicles = 15
        alloted_time = (total_vehicles/max_vehicles) * max_seconds
        alloted_time = min(alloted_time,max_seconds)
        prediction = alloted_time
        print("alloted time",alloted_time)

        print('/static/'+filename)
    return render_template('predict.html', alloted = prediction,ur='/static/outputs/'+filename)


@app.route('/',methods=['GET'])
def index():
    return render_template('myproject.html')



app.run()
