import cv2 as cv
import numpy as np
import pytesseract
from ShapeDetection import determine_shape_type
def preprocess_gray_smooth_threshold(image):
    grayScaleImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    smoothImage = cv.bilateralFilter(grayScaleImage,5,15,15,cv.BORDER_WRAP)
    thresholdedImage = cv.adaptiveThreshold(smoothImage,235,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,3)

    return thresholdedImage

def preprocess_gray_nlmeans_threshold(image):
    grayscaleImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    denoisedImage = cv.fastNlMeansDenoising(grayscaleImage,None,10,7,21)
    thresholdedImage = cv.adaptiveThreshold(denoisedImage,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,3)

    return thresholdedImage
def points_close_enough(pointSet1,pointSet2):
    if (pointSet1[0] - pointSet2[0]) > 4:
        return False
    if (pointSet1[1] - pointSet2[1]) > 3:
        return False
    if (pointSet1[2]-pointSet2[2]) > 4:
        return False
    if (pointSet1[3] - pointSet2[3]) > 4:
        return False
    return True


def getSkewAngle2(cvImage,upperArea,LowerArea,lowest_point_rect = 0):

    newImage = cvImage.copy()
    threshold = preprocess_gray_nlmeans_threshold(newImage)

    contours, hierarchy = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    target_contour = []
    for c in contours:
        rect = cv.boundingRect(c)
        x, y, w, h = rect

        if ((w * h) < (upperArea)) & ((w * h) > (LowerArea)):
                target_contour.append(c)
    x,y,w,h = cv.boundingRect(target_contour[0])
    area_max = cv.contourArea(target_contour[0])
    minimum_area_rect = cv.minAreaRect(target_contour[0])
    box_min = cv.boxPoints(minimum_area_rect)
    box_min = np.int32(box_min)
    area_min = cv.contourArea(box_min,True)
    print(area_min)
    print(area_max)
    if (area_min-area_max) < 0.01:
        print("yes")
        return newImage
    else:
        print("no")

        M = cv.getRotationMatrix2D((newImage.shape[0]/2,newImage.shape[1]/2),1,1.0)
        newImage = cv.warpAffine(cvImage,M,(cvImage.shape[0],cvImage.shape[1]))
        getSkewAngle2(newImage,upperArea+100,LowerArea-100,0)









def rotate_image2(image):
    print('d')
    newImage = getSkewAngle2(image,425,419)
    return newImage

def draw_boxes_OCR_output(image,height):
    boxes = pytesseract.image_to_boxes(image,
                                       config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM0987654321',
                                       timeout=2000)
    print(boxes)
    for b in boxes.splitlines():
        b = b.split(" ")
        cv.rectangle(
            image,
            (int(b[1]), height - int(b[2])),
            (int(b[3]), height - int(b[4])),
            (0, 255, 0),
            2,
        )
    cv.imshow('Boxes', image)

def rotate_180(image,centre):
    (h,w) = image.shape[:2]
    M = cv.getRotationMatrix2D(centre,180,1.0)
    newImage = cv.warpAffine(image,M,(h,w))
    return newImage

compImage = cv.imread('RealWork/Red_Cross_White_9.jpeg')
resized_image = cv.resize(compImage, (500, 500), interpolation=cv.INTER_LINEAR)
rotated_image = rotate_image2(resized_image)

cv.imshow('Rotated',rotated_image)
"""data = pytesseract.image_to_data(rotated_image,output_type=dict,config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM0987654321')
confidence1 = data['conf']
rotated_image_180 = rotate_180(rotated_image)
data2 = pytesseract.image_to_data(rotated_image_180,output_type=dict,config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVB')
confidence2 = data2['conf']
if confidence1>confidence2:

    print(pytesseract.image_to_string(rotated_image,config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM09876054321',timeout=2000))
    h, w = rotated_image.shape[:2]
    draw_boxes_OCR_output(rotated_image,h)
if confidence2 > confidence1:
    print(pytesseract.image_to_string(rotated_image_180,
                                      config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM09876054321',
                                      timeout=2000))
    h, w = rotated_image_180.shape[:2]
    draw_boxes_OCR_output(rotated_image_180, h)"""

cv.waitKey(0)
