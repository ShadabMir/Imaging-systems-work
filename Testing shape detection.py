import cv2 as cv
import numpy as np
import pytesseract
from ShapeDetection import determine_shape_type

def preprocess_gray_nlmeans_threshold(image):
    grayscaleImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    denoisedImage = cv.fastNlMeansDenoising(grayscaleImage,None,10,3,11)
    thresholdedImage = cv.adaptiveThreshold(denoisedImage,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,13,3)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    morphedImage = cv.morphologyEx(thresholdedImage,cv.MORPH_CLOSE,kernel,iterations=3)

    return morphedImage

analysisImage1 = cv.imread("Red_Pentagon_Green_N.jpeg")
analysisImage2 = cv.resize(analysisImage1,(500,500),interpolation=cv.INTER_LINEAR)

preprocessedImage = preprocess_gray_nlmeans_threshold(analysisImage2)
cv.imshow('Pre-processed Image',preprocessedImage)

contours,hierarchy = cv.findContours(preprocessedImage,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
sorted(contours,key=cv.contourArea)


contoursWithsShape = []
nameofShapes = []
i = 0
croppedImage = []
for c in contours:
    i += 1
    x,y,w,h = cv.boundingRect(c)
    minContour = cv.minAreaRect(c)
    box1 = cv.boxPoints(minContour)
    box = np.int32(box1)
    preprocessedImage2 = preprocessedImage.copy()
    cv.drawContours(preprocessedImage2,[box],0,(0,255,0),2)



    try:
        type, approx = determine_shape_type(c, preprocessedImage)

        if type == "UNKNOWN":
            y = 0

        else:
            contoursWithsShape.append(c)
            nameofShapes.append(type)

            """cv.drawContours(preprocessedImage,[c],0,(0,255,0),2)
            cv.imshow('Rec2t',preprocessedImage)"""

    except:
        x = 0

print(len(contoursWithsShape))

for c2 in contoursWithsShape:
    x,y,w,h = cv.boundingRect(c2)
    rectImage = cv.rectangle(preprocessedImage,(x,y),(x+w,y+h),(10,255,0),2)
    cv.imshow("Image",rectImage)
print(nameofShapes)


"""osd = pytesseract.image_to_osd(image)
angle = re.search('(?<=Rotate: )\d+', osd).group(0)"""

cv.waitKey(0)


