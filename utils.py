import cv2
import numpy as np

VERBOSE = 1


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

def findDimensions(image, homography):
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


def round_homography(H):
    """
    Usually, the homography produced is close, but might have errors.
    E.g. If taking two screen shorts of a moved window, the homography
    should be affine and without rotation.
    We use this function to correct the homography for these very common use cases.
    """

    H = H / H[2, 2]     # Make sure the homography is normalized

    # If h31 = h32 = 0 are close to 0 and h33 = 1, then we have an affine homography.
    # We use zero-based indices, so H[2,0], H[2,1] and H[2,2]
    if max(H[2,0], H[2,1]) < 1e-5 and H[2,2] > (1-1e-4):
        H[2,0] = H[2,1] = 0
        H[2,2] = 1
        print("Closest img homography is affine.")
        is_affine = True

    # Homogeneous point in 2D plane: (x1, x2, x3), where x=x1/x3, y=x2/x3. Typically (x, y, 1).
    # Isometry (preserves Euclidian distances):
    # [[R, t] [O, 1]]
    # where R is 2x2 rotation matrix, t is 2x1 translation vector, O is zeros.
    # If R is close to [[1,0], [0,1]], then there is no rotation or scaling.
    # If R is close to [[s,0], [0,s]], then there is scaling but no rotation.
    if np.isclose(H[0, 0], H[1, 1], rtol=1e-4, atol=1e-4) \
        and np.isclose(H[0, 1], 0, rtol=1e-4, atol=1e-4)\
        and np.isclose(H[1, 0], 0, rtol=1e-4, atol=1e-4):
        # We have no rotation:
        print("Closest img homography is un-rotated.")
        H[0, 1] = H[1, 0] = 0
        if np.isclose(H[0, 0], 1, rtol=1e-4, atol=1e-4):
            # Is isometric, since H[1,0]==H[0,1]==0:
            H[0, 0] = H[1, 1] = 1
            print("Closest img homography is isometric.")
        else:
            H[0, 0] = H[1, 1] = sum((H[0, 0], H[1, 1]))/2
    # Might still be isometric, if H[0:2,0:2] describes a rotation matrix...
    # This will be the case if H[0,1] == -H[1,0] and H[0,0] == -H[1,1] and
    # sqrt(H[0,0]**2 + H[1,0]**2) == 1
    return H


def combine_closest_image(base_img_rgb, closestImage, H_inv):

    # Determine if closestImg will enlarge the final, combined image:
    (min_x, min_y, max_x, max_y) = findDimensions(closestImage['img'], H_inv)
    # Adjust max_x and max_y by base img size
    max_x = max(max_x, base_img_rgb.shape[1])
    max_y = max(max_y, base_img_rgb.shape[0])

    # Translation part of the homography matrix:
    move_h = np.matrix(np.identity(3), np.float32) # 3x3 matrix
    if min_x < 0:
        # if closestImg will enlarge the combined image to the left.
        move_h[0, 2] += -min_x
        max_x += -min_x
    if min_y < 0:
        # if closestImg will enlarge the combined image to the top.
        move_h[1, 2] += -min_y
        max_y += -min_y

    mod_inv_h = move_h * H_inv

    # Attempting to eliminate the thin black border:
    img_w = int(max_x) # int(math.ceil(max_x))
    img_h = int(max_y) # int(math.ceil(max_y))

    if VERBOSE:
        print("Min Points: ", (min_x, min_y))
        print("Max Points: ", (max_x, max_y))
        print("New Dimensions: ", (img_w, img_h))

    # Warp the new image given the homography from the old image
    # Strictly speaking, you could just use warpAffine providing move_h as a 2x3 transformation matrix
    # http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html
    base_img_warp = cv2.warpPerspective(base_img_rgb, move_h, (img_w, img_h))

    # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
    # cv2.destroyAllWindows()

    next_img_warp = cv2.warpPerspective(closestImage['rgb'], mod_inv_h, (img_w, img_h))

    # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
    # cv2.destroyAllWindows()

    # Put the base image on an enlarged palette
    # TODO: Is this really the best way to do this?
    # How about just expanding the image by the required amount?
    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

    if VERBOSE:
        print("Enlarged Image Shape: ", enlarged_base_img.shape)
        print("Base Image Shape: ", base_img_rgb.shape)
        print("Base Image Warp Shape: ", base_img_warp.shape)

    enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, dtype=cv2.CV_8U)

    final_img = cv2.max(enlarged_base_img, next_img_warp)
    return final_img


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
