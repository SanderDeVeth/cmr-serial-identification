import cv2.cv2 as cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 30, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 40, 200, nothing)
cv2.createTrackbar("L - V", "Trackbars", 70, 200, nothing)
cv2.createTrackbar("U - H", "Trackbars", 80, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    hsv_mask = cv2.inRange(hsv, lower, upper)
    filtered = cv2.bitwise_and(frame, frame, mask=hsv_mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", hsv_mask)
    cv2.imshow("filtered", filtered)
    # cv2.imshow("img", img)
    # cv2.imshow("box", predMask)
    # cv2.imshow("hsv", hsv)
    # cv2.imshow("new image", new_img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()