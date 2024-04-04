import cv2
import numpy as np
import argparse
import matplotlib
import time
import os
from gtts import gTTS
#from playsound import playsound
import pygame
pygame.mixer.init()
score=0




# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)
# Load class Lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects Lists")
print(classes)

# Initialize camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Full HD


button_person = False
def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        polygon = np.array([(20, 20), (220, 20), (220, 70), (20, 70)])
        #cv2.fillPoly(frame, polygon, (0, 0, 200))
        #cv2.putText(frame, "Detect", (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        is_inside = cv2.pointPolygonTest(polygon, (x, y), False)
        if is_inside > 0:
            print("you are clicking in the button")
            if button_person is False:
                button_person = True
            else:
                button_person = False
            print("Now button person is: ", button_person)


# Create Window
cv2.namedWindow('Frame')
cv2.setMouseCallback("Frame", click_button)

while True:
    ret, frame = cam.read()
#    cv2.imshow("Frame", frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

        # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        #color = colors[class_id]
        print(class_name)
        print(score)
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 3)

    if button_person is True and not pygame.mixer.music.get_busy() and class_name == class_name and score>.5:
        score += 1
        name = class_name + ".mp3"


        if not os.path.isfile(name):
            tts = gTTS(text="I see a " + class_name , lang='en', slow=True)
            tts.save(name)

        last = 0
        pygame.mixer.music.load(name)
        pygame.mixer.music.play()

    # Create button
    cv2.rectangle(frame, (20, 20), (220, 70), (0, 0, 0), -1)
    polygon = np.array([(20, 20), (220, 20), (220, 70), (20, 70)])
    cv2.putText(frame, "iSense", (350, 350), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    #cv2.fillPoly(frame, polygon, (10, 10, 200))
    cv2.waitKey(1)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)












