#!/usr/bin/python

from __future__ import print_function, division
import os
import sys
import cv2
import math
import numpy as np
import utils

from numpy import linalg

VERBOSE = 1


class AlignImagesRansac(object):

    def __init__(self, image_dir, key_frame, output_dir, img_filter=None,
                 score_max=0.5, score_min=0.1,
                 output_filetype="PNG"):
        '''
        image_dir: 'directory' containing all images
        key_frame: 'dir/name.jpg' of the base image
        output_dir: 'directory' where to save output images
        optional:
           img_filter = 'JPG'; None->Take all files.

        score_min: minimum score required for any image to be appended.
        score_max: If an image has a score higher than this, it is immediately appended,
                   (without checking whether another image has an even higher score).
        '''

        self.output_filetype = output_filetype
        self.score_max = score_max
        self.score_min = score_min
        self.show_images = False # Will show images as they are stitched together.
        self.key_frame_file = os.path.split(key_frame)[-1]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        elif not os.path.isdir(output_dir):
            raise RuntimeWarning("output_dir is not a directory: %s" % output_dir)
        self.output_dir = output_dir

        # Open the directory given in the arguments
        try:
            self.dir_list = os.listdir(image_dir)
        except OSError:
            print("Unable to open directory: %s" % image_dir)
            sys.exit(-1)

        if img_filter:
            # remove all files that doen't end with .[image_filter]
            # TODO: Use glob or fnmatch-based filtering instead.
            self.dir_list = [fn for fn in self.dir_list if fn.find(img_filter)]
        if 'Thumbs.db' in self.dir_list: #remove Thumbs.db (windows only)
            self.dir_list.remove('Thumbs.db')

        self.dir_list = [os.path.join(image_dir, fn) for fn in self.dir_list]
        # Filter out anything that is not a file:
        self.dir_list = [fpath for fpath in self.dir_list if os.path.isfile(fpath)]
        # Remove any keyframes files:
        self.dir_list = [fpath for fpath in self.dir_list if fpath != key_frame]
        if VERBOSE:
            print("dir_list:", self.dir_list)

        base_img_rgb = cv2.imread(key_frame)
        if base_img_rgb is None:
            raise IOError("%s doesn't exist" %key_frame)
        # utils.showImage(base_img_rgb, scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()

        self.detector = cv2.SIFT() # I believe this is re-usable

        final_img = self.stitchImages(base_img_rgb, 0)



    def stitchImages(self, base_img_rgb, round=0):
        """
        Recursively stitch images in self.dir_list together, starting with base_img_rgb.
        Finds the closest image match of the (remaining) files in self.dir_list and
        combines that image with base_img_rgb.
        """

        if not self.dir_list:
            print("Dir list is exhausted - all done.")
            return base_img_rgb

        base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        # Use the SIFT feature detector
        #detector = cv2.SIFT()

        # Find key points in base image for motion estimation
        base_features, base_descs = self.detector.detectAndCompute(base_img, None)

        if self.show_images:
            # Create new key point list
            # Ever heard of list comprehensions? No? OK.
            key_points = [(int(kp.pt[0]),int(kp.pt[1])) for kp in base_features]
            utils.showImage(base_img, key_points, scale=(0.2, 0.2), timeout=0)
            cv2.destroyAllWindows()

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
            # TODO: Maybe load the images once and then keep them in memory?

            if VERBOSE > 1:
                print("Reading next image: %s ..." % next_img_path)

            if self.key_frame_file in next_img_path:
                print("\t Skipping keyframe image %s..." % self.key_frame_file)
                continue

            # Read in the next image...
            next_img_rgb = cv2.imread(next_img_path)
            next_img = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

            # Find points in the next frame
            next_features, next_descs = self.detector.detectAndCompute(next_img, None)
            matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)
            matches_subset = utils.filter_matches(matches)
            distance = utils.imageDistance(matches_subset)
            averagePointDistance = distance/float(len(matches_subset))
            kp1, kp2 = zip(*((base_features[match.trainIdx], next_features[match.queryIdx]) for match in matches_subset))
            p1, p2 = [np.array([k.pt for k in kps]) for kps in (kp1, kp2)]
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            inlierRatio = float(np.sum(status)) / float(len(status))

            if VERBOSE > 1:
                print("\t Finding points...")
                print("\t Match Count: ", len(matches))
                print("\t Filtered Match Count: ", len(matches_subset))
                print("\t Distance from Key Image: ", distance)
                print("\t Average Distance: ", averagePointDistance)
                print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

            # What is the best scoring criteria? inlierRatio or averagePointDistance?
            if closestImage is None or inlierRatio > closestImage['inliers']:
                closestImage = {'h': H,
                                'inliers': inlierRatio,
                                'dist': averagePointDistance,
                                'path': next_img_path,
                                'rgb': next_img_rgb,
                                'img': next_img,
                                'feat': next_features,
                                'desc': next_descs,
                                'match': matches_subset
                               }

        print("Closest Image: ", closestImage['path'])
        print("Closest Image Ratio: ", closestImage['inliers'])
        self.dir_list = [x for x in self.dir_list if x != closestImage['path']]
        if closestImage['inliers'] < self.score_min:
            # No suitable candidates...
            # Uh... if we didn't find any candidates here, we're not gonna find it in any other rounds either.
            # Probably just abort and return?
            print("Closest image doesn't pass criteria. Stopping here...")
            return base_img_rgb


        # utils.showImage(closestImage['img'], scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()

        # Calculate inverse homography matrix:
        H = closestImage['h']
        H = H / H[2, 2]         # Normalize
        H_inv = linalg.inv(H)   # TODO: Probably also round this one

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

        # Determine if closestImg will enlarge the final, combined image:
        (min_x, min_y, max_x, max_y) = utils.findDimensions(closestImage['img'], H_inv)
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
        # How about just expanding the image by the required amount?
        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

        print("Enlarged Image Shape: ", enlarged_base_img.shape)
        print("Base Image Shape: ", base_img_rgb.shape)
        print("Base Image Warp Shape: ", base_img_warp.shape)

        enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, dtype=cv2.CV_8U)

        final_img = cv2.max(enlarged_base_img, next_img_warp)

        if self.show_images:
            utils.showImage(final_img, scale=(0.2, 0.2), timeout=0)
            cv2.destroyAllWindows()

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

                if self.show_images:
                    utils.showImage(final_img_crop, scale=(0.2, 0.2), timeout=0)
                    cv2.destroyAllWindows()

                final_img = final_img_crop

        # Write out the current round
        final_filename = "%s/%d.%s" % (self.output_dir, round, self.output_filetype)
        cv2.imwrite(final_filename, final_img)

        return self.stitchImages(final_img, round+1)






 # ----------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: %s <image_dir> <key_frame> <output>" % sys.argv[0])
        sys.exit(-1)
    print("sys.argv:", sys.argv[1:])
    AlignImagesRansac(*sys.argv[1:])
