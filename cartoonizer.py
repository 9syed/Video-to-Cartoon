import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

width = 800
height = 600

cv2.namedWindow('Cartoonized', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Cartoonized', width, height)

while(True):
    ret, img = cap.read()
    X = img.reshape((-1, 3))
    X = np.float32(X)

    K = 12
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 0.9)
    ret, label, center = cv2.kmeans(X, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res2 = res2 + 15

    cv2.imshow('Cartoonized', res2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()