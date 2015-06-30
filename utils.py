import cv2
import numpy as np

def filter_matches(matches, ratio=0.75):
    filtered_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])

    return filtered_matches

def imageDistance(matches):

    sumDistance = 0.0

    for match in matches:

        sumDistance += match.distance

    return sumDistance

def findDimensions(self, image, homography):
    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0, 0]
    base_p2[:2] = [x, 0]
    base_p3[:2] = [0, y]
    base_p4[:2] = [x, y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

        if max_x == None or normal_pt[0, 0] > max_x:
            max_x = normal_pt[0, 0]

        if max_y == None or normal_pt[1, 0] > max_y:
            max_y = normal_pt[1, 0]

        if min_x == None or normal_pt[0, 0] < min_x:
            min_x = normal_pt[0, 0]

        if min_y == None or normal_pt[1, 0] < min_y:
            min_y = normal_pt[1, 0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)


def showImage(img1,
    pts=None,
    scale=None,
    title="Image", timeout=1000):

    # # Show me the tracked points!
    if ( len(img1.shape) == 3 ):
        composite_image = np.zeros((img1.shape[0], img1.shape[1], 3), np.uint8)
    else:
        composite_image = np.zeros((img1.shape[0], img1.shape[1]), np.uint8)

    composite_image[:,0:img1.shape[1]] += img1

    if ( pts != None ):
        for pt in pts:
            cv2.circle(composite_image, (pt[0], pt[1]), 5, 255)

    if ( scale != None ):
        fx = scale[0]
        fy = scale[1]

        composite_image = cv2.resize(composite_image, (0,0), fx=fx, fy=fy)

    cv2.imshow(title, composite_image);
    cv2.waitKey(timeout)

def showComposite(img1, img2,
    pts1=None, pts2=None,
    scale=None,
    title="Image", timeout=1000):

    # # Show me the tracked points!
    composite_image = np.zeros((img1.shape[0], img1.shape[1]*2), np.uint8)
    composite_image[:,0:img1.shape[1]] += img1
    composite_image[:,img1.shape[1]:img2.shape[1]*2] += img2

    if ( pts1 != None ):
        for pt in pts1:
            cv2.circle(composite_image, (pt[0], pt[1]), 5, 255)

    if ( pts2 != None ):
        for pt in pts2:
            x = int(pt[0]) + img1.shape[1]
            y = int(pt[1])
            cv2.circle(composite_image, (x, y), 5, 255)

    if ( scale != None ):
        fx = scale[0]
        fy = scale[1]

        composite_image = cv2.resize(composite_image, (0,0), fx=fx, fy=fy)

    cv2.imshow(title, composite_image);
    cv2.waitKey(timeout)
