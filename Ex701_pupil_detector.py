#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhage                    -->
#<!--                      Computer Science Department                      -->
#<!--                    Eye Information Research Group                     -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : Ex701_pupil_detector.py                                  -->
#<!-- Description: Script to detect pupils in eye images using binary       -->
#<!--            : images and blob detection                                -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : narcizo[at]itu[dot]dk                                    -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: No additional information                                -->
#<!-- Date       : 12/03/2018                                               -->
#<!-- Change     : 12/03/2018 - Creation of this script                     -->
#<!-- Review     : 12/03/2018 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018031201 $"

########################################################################
import cv2
import math
import numpy as np
import IAMLTools
import imutils

########################################################################

def onValuesChange(self, dummy=None):
    """ Handle updates when slides have changes."""
    global trackbarsValues
    trackbarsValues["threshold"] = cv2.getTrackbarPos("threshold", "Trackbars")
    trackbarsValues["minimum"]   = cv2.getTrackbarPos("minimum", "Trackbars")
    trackbarsValues["maximum"]   = cv2.getTrackbarPos("maximum", "Trackbars")
    #trackbarsValues["area"]  = cv2.getTrackbarPos("area", "Trackbars")

def showDetectedPupil(image, threshold, ellipses=None, centers=None, bestPupilID=None):
    """"
    Given an image and some eye feature coordinates, show the processed image.
    """
    # Copy the input image.
    processed = image.copy()
    if (len(processed.shape) == 2):
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # Draw the eye feature.
    if (ellipses is not None and len(ellipses) > 0 and
        centers  is not None and len(centers) > 0):
        for pupil, center in zip(ellipses, centers):
            cv2.ellipse(processed, pupil, (0, 0, 255), 2)
            if center[0] != -1 and center[1] != -1:
                cv2.circle(processed, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)            

        # Draw the best pupil candidate:
        if (bestPupilID is not None and bestPupilID != -1):
            pupil  = ellipses[bestPupilID]
            center = centers[bestPupilID]

            cv2.ellipse(processed, pupil, (0, 255, 0), 2)
            if center[0] != -1 and center[1] != -1:
                cv2.circle(processed, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)     

    # Show the processed image.
    cv2.imshow("Detected Pupil", processed)

def showScatterPlot(xData, yData, bestPupilID):
    """Plot the correlation of two blob properties."""
    #<!--------------------------------------------------------------------------->
    #<!--                            YOUR CODE HERE                             -->
    #<!--------------------------------------------------------------------------->

    pass

    #<!--------------------------------------------------------------------------->
    #<!--                                                                       -->
    #<!--------------------------------------------------------------------------->

def detectPupil(image, threshold=2, minimum=5, maximum=50):
    """
    Given an image, return the coordinates of the pupil candidates.
    """
    # Create the output variable.
    bestPupilID = -1
    ellipses    = []
    centers     = []
    area        = []

    kernel  = np.ones((5,5),np.uint8)
    
    # Grayscale image.
    grayscale = image.copy()
    if len(grayscale.shape) == 3:
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY)

    # Define the minimum and maximum size of the detected blob.
    minimum = int(round(math.pi * math.pow(minimum, 2)))
    maximum = int(round(math.pi * math.pow(maximum, 2)))
    #Bilablur = cv2.blur(grayscale,(3,5))
    #gaussianBlur = cv2.GaussianBlur(grayscale,(3,5),0)
    #median = cv2.medianBlur(grayscale,5)    
    
    #filter2D = cv2.filter2D(median,-1,kernel)
    blur= cv2.bilateralFilter(grayscale,9,40,40)
    
    # Create a binary image.
    _, thres = cv2.threshold(blur, threshold, 255,cv2.THRESH_BINARY_INV)
    
    cls = cv2.morphologyEx(thres,cv2.MORPH_CLOSE,kernel, iterations = 1)
    dilation = cv2.dilate(cls,kernel,iterations = 1)
    #erosion = cv2.erode(thres, kernel, iterations = 2)
    
    # Show the threshould image.
    cv2.imshow("Threshold", dilation)
    
    # Find blobs in the input image.
    contours, _= cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #<!--------------------------------------------------------------------------->
    #<!--                            YOUR CODE HERE                             -->
    #<!--------------------------------------------------------------------------->

    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    
   
    for i, c in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
    
        minRect[i] = cv2.minAreaRect(c)        
        M = cv2.moments(c)
        
        if (M["m00"] == 0) :
            M["m00"] = 1   
            
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    
        if c.shape[0] > minimum and c.shape[0] < maximum:
           
                                                
            minEllipse[i] = cv2.fitEllipse(c)
                               
            cv2.ellipse(image, minEllipse[i], (0,0,255), 2)#desenhando a ellipse
            
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)# desenhando o circulo central da ellipse

                    
        
    #<!--------------------------------------------------------------------------->
    #<!--                                                                       -->
    #<!--------------------------------------------------------------------------->

    # Return the final result.
    return ellipses, centers, bestPupilID
 

# Define the trackbars.
trackbarsValues = {}
trackbarsValues["threshold"] = 2
trackbarsValues["minimum"]  = 5
trackbarsValues["maximum"]  = 50
#trackbarsValues["area"]  = 5

# Create an OpenCV window and some trackbars.
cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("threshold", "Trackbars",  0, 255, onValuesChange)
cv2.createTrackbar("minimum",   "Trackbars",  5,  40, onValuesChange)
cv2.createTrackbar("maximum",   "Trackbars", 50, 100, onValuesChange)
#cv2.createTrackbar("area",  "Trackbars",  5, 400, onValuesChange)

cv2.imshow("Trackbars", np.zeros((3, 500), np.uint8))

# Create a capture video object.
filename = "inputs/eye01.mov"
capture = cv2.VideoCapture(filename)

# This repetion will run while there is a new frame in the video file or
# while the user do not press the "q" (quit) keyboard button.
while True:
    # Capture frame-by-frame.
    retval, frame = capture.read()

    # Check if there is a valid frame.
    if not retval:
        # Restart the video.
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Get the detection parameters values.
    threshold = trackbarsValues["threshold"]
    minimum   = trackbarsValues["minimum"]
    maximum   = trackbarsValues["maximum"]
    #area  = trackbarsValues["area"]  
    
    # Pupil detection.
    ellipses, centers, bestPupilID = detectPupil(frame, threshold, minimum, maximum)

    # Show the detected pupils.
    showDetectedPupil(frame, threshold, ellipses, centers, bestPupilID)

    # Display the captured frame.
    #cv2.imshow("Eye Image", frame)
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

# When everything done, release the capture object.
capture.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
