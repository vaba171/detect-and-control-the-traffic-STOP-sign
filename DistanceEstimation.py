import cv2 as cv 
import numpy as np
import time

# Distance constants 
KNOWN_DISTANCE = 200 #mm
#PERSON_WIDTH = 16 #INCHES
#MOBILE_WIDTH = 3.0 #INCHES
STOP_WIDTH = 39 #mm
# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("yolo.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny-custom_v3_192_final.weights', 'yolov4-tiny-custom_v2.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(192, 192), scale=1/255, swapRB=True)
 
# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # STOP class id
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        #elif classid ==4:
            #data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = ((real_object_width * focal_length) / width_in_frame) - 160 # 160 la khoang cac tu camera den dau xe
    #distance = ((real_object_width * focal_length) / width_in_frame) # test khoang cach camera
    return distance

# reading the reference image from dir 
#ref_person = cv.imread('ReferenceImages/image14.png')
ref_STOP = cv.imread('STOP_200(1).jpg')

#mobile_data = object_detector(ref_mobile)
#mobile_width_in_rf = mobile_data[1][1]

#STOP_data = object_detector(ref_STOP)
#STOP_width_in_rf = STOP_data[0][1]
STOP_width_in_rf= 160

#print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")
print(f" STOP sign width in pixel: {STOP_width_in_rf}  ")
# finding focal length 
#focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

#focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_STOP = focal_length_finder(KNOWN_DISTANCE, STOP_WIDTH, STOP_width_in_rf)

cap = cv.VideoCapture('stop2.mp4')
#cap = cv.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    print (h, w)
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    # putting the FPS count on the frame
    cv.putText(frame, "fps =  ", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(frame, fps, (95, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # frame = cv.resize(frame,(192,192))
    data = object_detector(frame)
    for d in data:
        if d[0] =='STOP':
            distance = distance_finder(focal_STOP, STOP_WIDTH, d[1])
            x, y = d[2]
        #elif d[0] =='STOP':
         #   distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
          #  x, y = d[2]
        #cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        #cv.putText(frame, f'Dis: {round(distance,2)} mm', (x+5,y+13), FONTS, 0.48, GREEN, 2)
        #print(distance)

        if distance < 500:
            #cv.putText(frame, "Phanh gap", (400,320), FONTS,3, (255,0,0),2)
            print(f" Khoang cach can phanh = { distance }  ")



    cv.imshow('frame',frame)

    #print(data_list)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

