import cv2 as cv

#load images
image1 = cv.imread("paris1.jpg")
image2 = cv.imread("paris.jpg")

#convert to grayscale image
gray_scale1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray_scale2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

#initialize SIFT object
sift = cv.SIFT_create()

kp1, des1= sift.detectAndCompute(image1, None)
kp2, des2= sift.detectAndCompute(image2, None)

# initialize Brute force matching
bf = cv.BFMatcher()

matches = bf.knnMatch(des1, des2, k = 2)
    #sort the matches
good = []
for m,n in matches:
    if m.distance < .95*n.distance:
        good.append([m])
        a=len(good)
        percent=(a*100)/len(kp2)
        print("{} % similarity".format(percent))
        if percent >= 60.00:
            print('Match Found')
        if percent < 60.00:
            print('Match not Found')

matched_imge = cv.drawMatchesKnn(image1, kp1, image2, kp2, good, None,flags=2)

cv.imshow("Matching Images", matched_imge)
cv.waitKey(0)