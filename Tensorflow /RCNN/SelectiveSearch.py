import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print("check 1 complete")
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
print("check 2 complete")
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
print("check 3 complete")
contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("check 4 complete")
yellow_bounding_rects = [cv2.boundingRect(contour) for contour in contours]
print("check 5 complete")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
print("check 6 complete")
rects = ss.process()

print("Filter Check  complete")
filtered_rects = []
for (x, y, w, h) in rects:
    for (yellow_x, yellow_y, yellow_w, yellow_h) in yellow_bounding_rects:
        if x >= yellow_x and y >= yellow_y and x + w <= yellow_x + yellow_w and y + h <= yellow_y + yellow_h:
            filtered_rects.append((x, y, w, h))
            break

print("Boundary Box check  complete")
for (x, y, w, h) in filtered_rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

