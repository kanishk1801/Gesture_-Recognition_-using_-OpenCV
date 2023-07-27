#This is a Computer vision package that makes its easy to run Image processing and AI functions.
# At the core it uses OpenCV and Mediapipe libraries.
#Hand tracking is the process in which a computer uses computer vision to detect a hand from
# an input image and keeps focus on the hand’s movement and orientation.
#mediapipe and OpenCV libraries in python to detect Hand.
from cvzone.HandTrackingModule import HandDetector
#cv2 is the module import name for opencv-python,
import cv2
# provides the facility to establish the interaction between the user and the operating system.
import os
#Python library used for working with arrays.
import numpy as np


# Parameters
width, height = 720, 500
gestureThreshold = 400
folderPath = "presentation"



# Camera Setup
#VideoCapture(0)-opens default camera for video capturing
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)



# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 3), int(213 * 3)  # width and height of small image

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    #cap.read() returns a bool ( True / False ).
    # If the frame is read correctly, it will be True .
    success, img = cap.read()

    #cv2.flip() method is used to flip a 2D array.
    # The function cv::flip flips a 2D array around vertical, horizontal, or both axes.
    #flip code: A flag to specify how to flip the array;
    # 0 means flipping around the x-axis and positive value (for example, 1) means flipping around y-axis.
    # Negative value (for example, -1) means flipping around both axes.
    # Return Value: It returns an image.
    img = cv2.flip(img, 1)


    #os. path. join combines path names into one complete path.
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:  # If hand is detected

        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        #numpy.interp() function returns the one-dimensional piecewise linear interpolant to a function
        # with given discrete data points (xp, fp), evaluated at x.
        #Syntax : numpy.interp(x, xp, fp, left = None, right = None, period = None)

        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # If hand is at the height of the face
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        if fingers == [0, 1, 1, 0, 0]:
            #cv2.circle() method is used to draw a circle on any image.
            #cv2.circle(image, center_coordinates, radius, color, thickness)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                #cv2.line() method is used to draw a line on any image.
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    #To resize an image
    #cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    imgSmall = cv2.resize(img, (ws, hs))
    #get the current shape of an array,
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall


    #cv2.imshow() method is used to display an image in a window.
    # The window automatically fits the image size.
    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    # cv2 waitkey() allows you to wait for a specific time in milliseconds
    # until you press any button on the keyword
    key = cv2.waitKey(1)
    #ord() function takesstring argument of a single Unicode character
    # and return its integer Unicode code point value.
    if key == ord('q'):
        break