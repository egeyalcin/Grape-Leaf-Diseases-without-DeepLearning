from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

class classification:
    def show(self):
        global onlyfiles
        for n in range(len(onlyfiles)):
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            images = np.empty(len(onlyfiles[n]), dtype=object)
            images[n] = cv2.imread(join(mypath, onlyfiles[n]))
            hsv = cv2.cvtColor(images[n], cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, brown_lower, brown_upper)

            white_cell = np.sum(mask == 0)

            if white_cell < 55000:
                cv2.putText(images[n], "Isariopsis", (20, 40), font, 1, (0, 0, 0))
                cv2.imshow("img{}".format(n + 1), images[n])
                cv2.waitKey(0)

            if  white_cell  > 55000 and white_cell < 60000:
                cv2.putText(images[n], "Esca", (20, 60), font, 1, (0, 0, 0))
                cv2.imshow("img{}".format(n + 1), images[n])
                cv2.waitKey(0)

            if white_cell  > 60000:
                cv2.putText(images[n], "Blackrot", (20, 80), font, 1, (0, 0, 0))
                cv2.imshow("img{}".format(n + 1), images[n])
                cv2.waitKey(0)
            #cv2.destroyAllWindows()

mypath=r"D:\WORKSPACE\computer_vision\videos_and_images\grapeLeafs"
mypath_len=len(mypath)
mysavepath=r"D:\WORKSPACE\computer_vision\videos_and_images\grapeLeafs\*.JPG"
font=cv2.FONT_HERSHEY_SIMPLEX
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]) )

hsv = cv2.cvtColor(images[n], cv2.COLOR_BGR2HSV)

brown_lower=np.array([0,0,0])
brown_upper=np.array([15,255,150])

mask = cv2.inRange(hsv,brown_lower, brown_upper)
bitwise=cv2.bitwise_and(images[n],images[n],mask=mask)
contours,ret=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
zeros_image=np.zeros([images[n].shape[0],images[n].shape[1],1])

hull=[]

for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i],False))

for cnt in contours:
    area = cv2.contourArea(cnt)
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if area > 1600:
      cv2.drawContours(images[n], [approx], 0, (0, 0, 0), 5)
      if len(approx) > 3:
        cv2.putText(images[n], "Yaprak hastaliği", (x, y), font, 1, (0, 0, 0))

cagir = classification()
cagir.show()


