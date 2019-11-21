import cv2
import numpy as np

cap = cv2.VideoCapture('./videos/challenge.mp4')

def makeCoordinates(img, lineParaments):
    slope, intercept = lineParaments

    y1 = image.shape[0]
    y2 = int(y1 * ( 3 / 5 ))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = makeCoordinates(img, left_fit_average)
    right_line = makeCoordinates(img, right_fit_average)

    return np.array([left_line, right_line])

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def drawLines(img, lines):
    lineImg = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lineImg, (x1, y1), (x2, y2), (255, 0, 0,), 1)
    return lineImg

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height, (550, 250))]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

while (cap.isOpened()):
    _, frame = cap.read()
    
    cannyImg = canny(frame)
    croppedImg = region_of_interest(cannyImg)
    lines = cv2.HoughLinesP(croppedImg, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averageLines = average_slope_intercept(frame, lines)
    lineImg = drawLines(frame, averageLines)
    comboImg = cv2.addWeighted(frame, 0.8, lineImg, 1, 1)

    cv2.imshow("Result", comboImg)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
