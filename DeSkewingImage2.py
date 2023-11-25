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


def getSkewAngle2(cvImage,upperWidth,upperHeight,lowerheight,lowerwidth,lowest_point_rect = 0):

    newImage = cvImage.copy()
    resized_image = cv.resize(newImage, (500, 500), interpolation=cv.INTER_LINEAR)
    threshold = preprocess_gray_nlmeans_threshold(resized_image)

    contours, hierarchy = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    target_contour = []
    for c in contours:
        rect = cv.boundingRect(c)
        x, y, w, h = rect

        if ((w * h) < (upperWidth * upperHeight)) & ((w * h) > (lowerwidth * lowerheight)):

            if determine_shape_type(c) == 'UNKNOWM':
                print('x')
            else:

                target_contour.append(c)
    x,y,w,h = cv.boundingRect(target_contour[0])
    area_max = cv.contourArea(target_contour[0])
    minimum_area_rect = cv.minAreaRect(target_contour[0])
    box_min = cv.boxPoints(minimum_area_rect)
    box_min = np.int32(box_min)
    area_min = cv.contourArea(box_min,True)
    if (area_max-area_min) < 0.01:
        if points_close_enough:
            return newImage
    else:

        M = cv.getRotationMatrix2D((cvImage.shape[0]/2,cvImage.shape[1]/2),1,1.0)
        newImage = cv.warpAffine(cvImage,M,(cvImage.shape[0]/2,cvImage.shape[1]/2))
        getSkewAngle2(newImage,upperWidth,upperHeight,lowerheight,lowerwidth,0)





    pass

def rotate_image(cvImage,target,centre, angle: float):
    print('e')
    newImage = cvImage.copy()
    newImage = cv.resize(newImage.copy(),(500,500),interpolation=cv.INTER_LINEAR)
    (h, w) = (newImage.shape[0],newImage.shape[1])
    M = cv.getRotationMatrix2D(centre, angle, 0.8)
    newImage = cv.warpAffine(newImage, M, (h,w))
    cv.imshow('new image test',newImage)
    return newImage

def rotate_image2(image):
    print('d')
    newImage = getSkewAngle2(image)
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
    neeImage = cv.warpAffine(image,M,(h,w))
compImage = cv.imread('Red_Cross_White_9.jpeg')
rotated_image = rotate_image2(compImage)
rotated_image2 = preprocess_gray_smooth_threshold(rotated_image)
cv.imshow('Rotated',rotated_image2)
data = pytesseract.image_to_data(rotated_image2,output_type=dict,config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM0987654321')
confidence1 = data['conf']
rotated_image_180 = rotate_180(rotated_image2)
data2 = pytesseract.image_to_data(rotated_image_180,output_type=dict,config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVB')
confidence2 = data2['conf']
if confidence1>confidence2:

    print(pytesseract.image_to_string(rotated_image2,config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM09876054321',timeout=2000))
    h, w = rotated_image2.shape[:2]
    draw_boxes_OCR_output(rotated_image2,h)
if confidence2 > confidence1:
    print(pytesseract.image_to_string(rotated_image_180,
                                      config='--psm 10 --oem 3-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM09876054321',
                                      timeout=2000))
    h, w = rotated_image_180.shape[:2]
    draw_boxes_OCR_output(rotated_image_180, h)

cv.waitKey(0)
