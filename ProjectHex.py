import cv2 as cv
import numpy as np
import pytesseract
import PIL


tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = tesseract_path
def preprocess_gray_nlmeans_threshold(image):
    grayscaleImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    denoisedImage = cv.fastNlMeansDenoising(grayscaleImage,None,10,7,21)
    thresholdedImage = cv.adaptiveThreshold(denoisedImage,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,3)

    return thresholdedImage
cam = cv.VideoCapture(0)
key_in = cv.waitKey(1)
cap_count = 0
while cam.isOpened():
    key_in_last = key_in
    key_in = cv.waitKey(1)
    ret, image = cam.read()
    if not ret:
        print("Cant recieve frame")
        break
    elif key_in == ord("p"):
        print("Quit pressed,Quitting")
        break
    processed_image = preprocess_gray_nlmeans_threshold(image)
    contours,hierarchy = cv.findContours(processed_image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for c in contours:

        x,y,w,h = cv.boundingRect(c)
        area = image.shape[0] * image.shape[1]
        if (w*h)< area/4 and (w*h)>area/8:
            cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow("Live feed", image)
    if key_in == ord(" "):
        cap_count += 1
        gray1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.bilateralFilter(gray1,5,15,15,cv.BORDER_DEFAULT)
        threshold_image = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10
        ) # Applys different threshold values to different blocks of the image to account for lighting differencesp

        erode = cv.morphologyEx(threshold_image, cv.MORPH_CLOSE, (3, 3), iterations=1)  # Works on Binary Image, Dialates and erodes together - Makes black noise large then small and white noise small and then big(the noises will dissapear with the reduction in size)
        cv.imshow(f"Capture {cap_count}", erode)
        print(pytesseract.image_to_string(erode, config="--psm 10 --oem 3"))
        h, w = image.shape[:2]

        data = pytesseract.image_to_boxes(erode, config="--psm 10 --oem 3")
        print(data)
        print(len(data.splitlines()))
        for b in data.splitlines():
            b = b.split(" ")
            cv.rectangle(
                image,
                (int(b[1]), h - int(b[2])),
                (int(b[3]), h - int(b[4])),
                (0, 255, 0),
                2,
            )
        cv.imshow("Image", image)

        """contour = pytesseract.image_to_boxes(erode,config='--psm 10 --oem 3 ')"""

        """contours,hierarchy = cv.findContours(erode,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours,key= cv.contourArea)
        rect = cv.boundingRect(contour[-1])
        x,y,w,h = rect
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv.imshow('Contours',image)"""

cam.release()
