import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 720) #width
cap.set(4, 480) #height
cap.set(10, 70) #brightness

classNames = []
classFile = 'coco_reformatted.names' #contains the formatted class names from COCO datset
with open(classFile, 'rt') as f:  #readTextmode
    classNames = f.read().rstrip('\n').split('\n')  #split strings and space

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (720, 480))
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)  #detection probabilty
    print(classIds, bbox)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('output', img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
