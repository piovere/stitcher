#!/usr/bin/python

from __future__ import print_function, division
import os
import sys
import cv2
import math
import numpy as np
#import utils

from numpy import linalg

VERBOSE = 1


class AlignImagesRansac(object):

    def __init__(self, image_dir, key_frame, output_dir, img_filter=None):
        '''
        image_dir: 'directory' containing all images
        key_frame: 'dir/name.jpg' of the base image
        output_dir: 'directory' where to save output images
        optional:
           img_filter = 'JPG'; None->Take all images
        '''

        self.key_frame_file = os.path.split(key_frame)[-1]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        elif not os.path.isdir(output_dir):
            raise RuntimeWarning("output_dir is not a directory: %s" % output_dir)
        self.output_dir = output_dir


        # Open the directory given in the arguments
        self.dir_list = []
        try:
            self.dir_list = os.listdir(image_dir)
            if img_filter:
                # remove all files that doen't end with .[image_filter]
                self.dir_list = filter(lambda x: x.find(img_filter) > -1, self.dir_list)
            try: #remove Thumbs.db, is existent (windows only)
                self.dir_list.remove('Thumbs.db')
            except ValueError:
                pass


        except OSError:
            print("Unable to open directory: %s" % image_dir)
            sys.exit(-1)

        self.dir_list = map(lambda x: os.path.join(image_dir, x), self.dir_list)

        self.dir_list = filter(lambda x: x != key_frame, self.dir_list)
        self.dir_list = list(self.dir_list)
        print("dir_list:", self.dir_list)

        base_img_rgb = cv2.imread(key_frame)
        if base_img_rgb is None:
            raise IOError("%s doesn't exist" %key_frame)
        # utils.showImage(base_img_rgb, scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()

        final_img = self.stitchImages(base_img_rgb, 0)



    def filter_matches(self, matches, ratio=0.75):
        filtered_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])

        return filtered_matches

    def imageDistance(self, matches):

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


    def stitchImages(self, base_img_rgb, round=0):

        if len(self.dir_list) < 1:
            return base_img_rgb

       # print base_img_rgb.channels()
    # if(image.channels()==1)
    # {      /* Grayscale */                                             }
    # else if (image.channels==4)
    # {      /* ARGB or RGBA image */


        base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        # Use the SIFT feature detector
        detector = cv2.SIFT()

        # Find key points in base image for motion estimation
        base_features, base_descs = detector.detectAndCompute(base_img, None)

        # Create new key point list
        # key_points = []
        # for kp in base_features:
        #     key_points.append((int(kp.pt[0]),int(kp.pt[1])))
        # utils.showImage(base_img, key_points, scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()

        # Parameters for nearest-neighbor matching
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        matcher = cv2.FlannBasedMatcher(flann_params, {})

        print("Iterating through next images...")

        closestImage = None

        # TODO: Thread this loop since each iteration is independent

        # Find the best next image from the remaining images
        for next_img_path in self.dir_list:

            # TODO: Optimization. If we find an image that is "good enough", perhaps
            # just use this, without going through the whole list.
            # For instance, if the inliers fraction is > 0.50 then it is a pretty good match.
            # In fact, even > 0.30 is pretty good.
            # Alternatively, as you go through the whole list, save the scores.
            # Then at the end, start by re-calculating only for a subset of the
            # images above a certain threshold.
            # This will postpone calculation of distant/unrelevant images,
            # using initial calculations to focus on good candidates.
            # You could do both - and also sort the images by their similarity score.
            # TODO: Refactor this into separate "find candidate" and "merge images" functions.

            if VERBOSE > 1:
                print("Reading next image: %s ..." % next_img_path)

            if self.key_frame_file in next_img_path:
                print("\t Skipping %s..." % self.key_frame_file)
                continue

            # Read in the next image...
            next_img_rgb = cv2.imread(next_img_path)
            next_img = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

            # if ( next_img.shape != base_img.shape ):
            #     print("\t Skipping %s, bad shape: %s" % (next_img_path, next_img.shape)
            #     continue

            if VERBOSE > 1:
                print("\t Finding points...")

            # Find points in the next frame
            next_features, next_descs = detector.detectAndCompute(next_img, None)

            matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)

            if VERBOSE > 1:
                print("\t Match Count: ", len(matches))

            matches_subset = self.filter_matches(matches)

            if VERBOSE > 1:
                print("\t Filtered Match Count: ", len(matches_subset))

            distance = self.imageDistance(matches_subset)

            if VERBOSE > 1:
                print("\t Distance from Key Image: ", distance)

            averagePointDistance = distance/float(len(matches_subset))

            if VERBOSE > 1:
                print("\t Average Distance: ", averagePointDistance)

            #kp1 = []
            #kp2 = []
            #
            #for match in matches_subset:
            #    kp1.append(base_features[match.trainIdx])
            #    kp2.append(next_features[match.queryIdx])

            kp1, kp2 = zip(*((base_features[match.trainIdx], next_features[match.queryIdx]) for match in matches_subset))

            #p1 = np.array([k.pt for k in kp1])
            #p2 = np.array([k.pt for k in kp2])
            p1, p2 = [np.array([k.pt for k in kps]) for kps in (kp1, kp2)]

            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            if VERBOSE > 1:
                print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

            inlierRatio = float(np.sum(status)) / float(len(status))

            # if ( closestImage == None or averagePointDistance < closestImage['dist'] ):
            if closestImage == None or inlierRatio > closestImage['inliers']:
                closestImage = {}
                closestImage['h'] = H
                closestImage['inliers'] = inlierRatio
                closestImage['dist'] = averagePointDistance
                closestImage['path'] = next_img_path
                closestImage['rgb'] = next_img_rgb
                closestImage['img'] = next_img
                closestImage['feat'] = next_features
                closestImage['desc'] = next_descs
                closestImage['match'] = matches_subset

        print("Closest Image: ", closestImage['path'])
        print("Closest Image Ratio: ", closestImage['inliers'])
        self.dir_list = [x for x in self.dir_list if x != closestImage['path']]
        if closestImage['inliers'] < 0.1:
            # No suitable candidates...
            # return self.stitchImages(base_img_rgb, round+1)
            # Uh... if we didn't find any candidates here, we're not gonna find it in any other rounds either.
            # Probably just abort and return?
            print("Closest image doesn't pass criteria. Stopping here...")
            return base_img_rgb


        # utils.showImage(closestImage['img'], scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()

        # Calculate inverse homography matrix:
        H = closestImage['h']
        H = H / H[2, 2]         # Normalize
        H_inv = linalg.inv(H)

        # If h31 = h32 = 0 are close to 0 and h33 = 1, then we have an affine homography.
        # We use zero-based indices, so H[2,0], H[2,1] and H[2,2]
        if max(H[2,0], H[2,1]) < 1e-5 and H[2,2] > (1-1e-4):
            H[2,0] = H[2,1] = 0
            H[2,2] = 1
            print("Closest img homography is affine.")
            is_affine = True

        # Homogeneous point in 2D plane: (x1, x2, x3), where x=x1/x3, y=x2/x3. Typically (x, y, 1).
        # Isometry (preserves Euclidian distances):
        # [[R, t],
        #   O, 1]]
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

        # Determine if closestImg will enlarge the final, combined image:
        (min_x, min_y, max_x, max_y) = self.findDimensions(closestImage['img'], H_inv)
        # Adjust max_x and max_y by base img size
        max_x = max(max_x, base_img.shape[1])
        max_y = max(max_y, base_img.shape[0])

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

        print("Homography: \n", H)
        print("Inverse Homography: \n", H_inv)
        print("Min Points: ", (min_x, min_y))
        print("Max Points: ", (max_x, max_y))

        mod_inv_h = move_h * H_inv

        # Attempting to eliminate the thin black border:
        img_w = int(max_x) # int(math.ceil(max_x))
        img_h = int(max_y) # int(math.ceil(max_y))
        print("img_w, img_h vs math.ceil(max_x, max_y): (%s, %s) vs (%s, %s)" % \
              (img_w, img_h, math.ceil(max_x), math.ceil(max_y)))

        print("New Dimensions: ", (img_w, img_h))

        # Warp the new image given the homography from the old image
        # Strictly speaking, you could just use warpAffine providing move_h as a 2x3 transformation matrix
        # http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html
        base_img_warp = cv2.warpPerspective(base_img_rgb, move_h, (img_w, img_h))
        print("Warped base image")

        # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
        # cv2.destroyAllWindows()

        next_img_warp = cv2.warpPerspective(closestImage['rgb'], mod_inv_h, (img_w, img_h))
        print("Warped next image")

        # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
        # cv2.destroyAllWindows()

        # Put the base image on an enlarged palette
        # TODO: Is this really the best way to do this?
        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

        print("Enlarged Image Shape: ", enlarged_base_img.shape)
        print("Base Image Shape: ", base_img_rgb.shape)
        print("Base Image Warp Shape: ", base_img_warp.shape)

        # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
        # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp

        # Create a mask from the warped image for constructing masked composite
        (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),
                                        0, 255, cv2.THRESH_BINARY)

        enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
                                    #mask=np.bitwise_not(data_map),
                                    dtype=cv2.CV_8U)

        #enlarged_base_img = cv2.addWeighted(enlarged_base_img, 1.0,
        #                                    base_img_warp, 0.5,
        #                                    gamma=0,
        #                                    mask=data_map,
        #                                    dtype=cv2.CV_8U)

        # Now add the warped image
        #final_img = cv2.add(enlarged_base_img, next_img_warp,
        #                    dtype=cv2.CV_8U)

        # Wouldn't it be simpler just to not have a mask and simply replace the pixels
        # of enlarged_base_img with the pixels from next_img_warp?
        # E.g. by using "lighten" image blend mode, instead of a mask:
        # np.maximum(enlarged_base_img, next_img_warp)
        # Alternatively, use OpenCV's per-element operations:
        # http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#max
        # http://docs.opencv.org/modules/gpu/doc/per_element_operations.html
        # If you want "fractional" blending, you have to first calculate the diff,
        # then zero-out negative values, then multiply with the fraction and then add.
        final_img = cv2.max(enlarged_base_img, next_img_warp)


        # utils.showImage(final_img, scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()

        eliminate_contours = False
        if eliminate_contours:
            # Crop off the black edges
            final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print("Found %d contours..." % (len(contours)))

            max_area = 0
            best_rect = (0, 0, 0, 0)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # print("Bounding Rectangle: ", (x,y,w,h)

                deltaHeight = h-y
                deltaWidth = w-x

                area = deltaHeight * deltaWidth

                if area > max_area and deltaHeight > 0 and deltaWidth > 0:
                    max_area = area
                    best_rect = (x, y, w, h)

            if max_area > 0:
                print("Maximum Contour: ", max_area)
                print("Best Rectangle: ", best_rect)

                final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
                                           best_rect[0]:best_rect[0]+best_rect[2]]

                # utils.showImage(final_img_crop, scale=(0.2, 0.2), timeout=0)
                # cv2.destroyAllWindows()

                final_img = final_img_crop

        # Write out the current round
        output_filetype = "PNG"
        final_filename = "%s/%d.%s" % (self.output_dir, round, output_filetype)
        cv2.imwrite(final_filename, final_img)

        return self.stitchImages(final_img, round+1)






 # ----------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: %s <image_dir> <key_frame> <output>" % sys.argv[0])
        sys.exit(-1)
    print("sys.argv:", sys.argv[1:])
    AlignImagesRansac(*sys.argv[1:])
