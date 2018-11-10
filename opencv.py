#import imutils
import cv2
image = cv2.imread("th.jpeg")
cv2.imshow("Original", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Gaussian Blur", gray)
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Threshold", thresh)
img, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, cnts, -1, (0, 255, 255), 2)



cv2.imshow("Contours", image)
cv2.waitKey(0)
