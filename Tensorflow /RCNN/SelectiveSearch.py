import cv2
import matplotlib.pyplot as plt

image_path = r"C:\Users\rohan\OneDrive\Desktop\A00_01.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Error loading image")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("check 1 complete")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    print("check 2  complete")
    ss.switchToSelectiveSearchFast()
    print("check 3 complete")
    rects = ss.process()
    print("check 4 complete")
    print('Total Number of Region Proposals: {}'.format(len(rects)))
    for (x, y, w, h) in rects:
        if w < 20 or h < 20 or w * h < 2000:
            continue
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print("check 5 complete")
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
