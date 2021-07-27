#All of the post and pre-processing of this model was done using openCV2
import cv2 as cv
#This package was used for mathematical operations
import numpy as np
#import pyttsx3
import math
#This package is used to convert text to speech
import gtts
#import os

from pygame import mixer

#initializing set to copy the items that get detected by the model
text1=set()

#engine = pyttsx3.init()

#Getting the XML file which contains human face contourts that has been created
# by training human positive and human negative images
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

distance = 0.0
#engine.setProperty('rate', 150)    
#engine.setProperty('volume', 0.9)  

#Initializing the minimum threshold to consider outcome
confThreshold = 0.25

# Non-Maximum threshold which is used to refine the bounding boxes, in this case any 
#bounding box that has less than 0.4 value is removed
nmsThreshold = 0.40

#Initializing the input width and height of the frames
inpWidth = 416
inpHeight = 416


classesFile = "coco.names"
classes = None

#Appending all the class names into a file which can be used on bounding boxes while displaying 
# the output
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#Initializing Configuration file
modelConf = 'yolov3.cfg'

# Intializing the weight file that has been trained using the coco dataset
modelWeights = 'yolov3.weights'


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []
    pred = " " 
    
# for each detetion from each output layer get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.25)

    for out in outs:
        for detection in out:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

#Checking if the confidence returned by the model is greater than the threshold and
# and mentioning the bounding box co-ordinates accordingly if the condition is satisfied
            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(centerX - width / 2)
                top = int(centerY - height / 2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
#Creating a user defined function to draw and label the bounding box using the co-ordinates
# that were calculated
        pred=drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)
    return pred

def drawPred(classId, conf, left, top, right, bottom):
    
#setting the frame
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
# setting the floating point for confidence of the label
    label = '%.2f' % conf
    
#If class is not that of a person, it is simply displayed    
    if classId>0:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        pred=(classes[classId])
        
# if the class label predicted is that of a person, we use the haar cascade trained contour 
# xml file to calculate the distance of human from the camera        
    if classId==0:
        print(classId)
        pred=(classes[classId])
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x, y, w, h) in faces:
#Formula used to calculate the distance of predicted human
            distancei = (2*3.14 * 180)/(w+h*360)*1000 + 3
            distance = math.floor(distancei/2)
            #print(distance)
            label= '%s:%s:%s' % (classes[classId], label, distance) + " inches"

 #This function is used to label the bounding box   
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    return pred

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



#returns the Net object which contains the network model.
net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
#To instruct network to use specific computation backend.
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#To instruct network to make computations on specific target device
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

winName = 'Object Detector'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

#Resizing the window
cv.resizeWindow(winName, 1000, 1000)

#Start capturing the video
cap = cv.VideoCapture(0)


while cv.waitKey(1) < 0:
    
    hasFrame, frame = cap.read()
    
    frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)

    
# Returns a blob with 4 dimension matrix which is our input image after mean subtraction, normalizing, 
# and channel swapping
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

#Setting the 4 dimensional matrix as input
    net.setInput(blob)
#Getting anf forwarding the unconnected outputs
    outs = net.forward(getOutputsNames(net))

    p=postprocess(frame, outs)
    
#Adding to the set, the class name that has been returned by the user defines function post propcess
    text1.add(p)
    

#To display the frame     
    cv.imshow(winName, frame)
    
#This user defined function adds all the set members into a string
    def convert(text1):
        new=""
        
        for x in text1:
            new += " "+x
        return new
    newtext=convert(text1)
    print(newtext)
    
# Google text to speech class here is used to load the output classes or predictions
# into an MP3 file
    tts=gtts.tts.gTTS(text=newtext, lang='en')
    tts.save("audiooutput3.mp3")
#Mixer is a pygame class which helps in dynamically loading the MP3 file and play it on the spot
    mixer.init()
    mixer.music.load('audiooutput3.mp3')
    mixer.music.play() 
    
    
     


cv.waitKey(0)
cv.destroyAllWindows()
