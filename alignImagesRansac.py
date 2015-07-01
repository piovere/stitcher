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
        self.eliminate_contours = False

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

        ## I believe feature detector and matcher are both re-usable: ##

        # Use the SIFT feature detector
        self.detector = cv2.SIFT()

        # Parameters for nearest-neighbor matching
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})

        final_img = self.stitchImages(base_img_rgb, 0)


    def calculate_homography_stats(self, base_features, base_descs, next_img_path):
        # Read in the next image...
        next_img_rgb = cv2.imread(next_img_path)
        next_img = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        # Find points in the next frame
        next_features, next_descs = self.detector.detectAndCompute(next_img, None)
        matches = self.matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)
        matches_subset = utils.filter_matches(matches)
        distance = utils.imageDistance(matches_subset)
        averagePointDistance = distance/len(matches_subset)
        kp1, kp2 = zip(*((base_features[match.trainIdx], next_features[match.queryIdx]) for match in matches_subset))
        p1, p2 = [np.array([k.pt for k in kps]) for kps in (kp1, kp2)]
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        inlierRatio = np.sum(status)/len(status)

        if VERBOSE > 1:
            print("Homography stats for next image candidate: %s ..." % next_img_path)
            print("\t Finding points...")
            print("\t Match Count: ", len(matches))
            print("\t Filtered Match Count: ", len(matches_subset))
            print("\t Distance from Key Image: ", distance)
            print("\t Average Distance: ", averagePointDistance)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

        stats = {'h': H,
                 'inliers': inlierRatio,
                 'dist': averagePointDistance,
                 'path': next_img_path,
                 'rgb': next_img_rgb,
                 'img': next_img,
                 'feat': next_features,
                 'desc': next_descs,
                 'match': matches_subset
                }
        return stats


    def find_closest_image(self, base_img):
        # Find key points in base image for motion estimation
        base_features, base_descs = self.detector.detectAndCompute(base_img, None)

        if self.show_images:
            # Create new key point list
            # Ever heard of list comprehensions? No? OK.
            key_points = [(int(kp.pt[0]),int(kp.pt[1])) for kp in base_features]
            utils.showImage(base_img, key_points, scale=(0.2, 0.2), timeout=0)
            cv2.destroyAllWindows()

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

            if self.key_frame_file in next_img_path:
                print("\t Skipping keyframe image %s..." % self.key_frame_file)
                continue
            homography_stats = self.calculate_homography_stats(base_features, base_descs, next_img_path)

            # What is the best scoring criteria? inlierRatio or averagePointDistance?
            if closestImage is None or homography_stats['inliers'] > closestImage['inliers']:
                closestImage = homography_stats
                if homography_stats['inliers'] > self.score_max:
                    print("%s is above score_max, combining immediately." % next_img_path)
                    break

        print("Closest Image: ", closestImage['path'])
        print("Closest Image Ratio: ", closestImage['inliers'])
        self.dir_list = [x for x in self.dir_list if x != closestImage['path']]
        return closestImage


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

        closestImage = self.find_closest_image(base_img)

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
        H = utils.round_homography(H)
        H_inv = linalg.inv(H)   # TODO: Probably also round this one
        if VERBOSE:
            print("Homography: \n", H)
            print("Inverse Homography: \n", H_inv)

        final_img = utils.combine_closest_image(base_img_rgb, closestImage, H_inv)

        if self.show_images:
            utils.showImage(final_img, scale=(0.2, 0.2), timeout=0)
            cv2.destroyAllWindows()

        if self.eliminate_contours:
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
