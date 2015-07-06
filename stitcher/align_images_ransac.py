#!/usr/bin/python

from __future__ import print_function, division, absolute_import
import os
import sys
from fnmatch import fnmatch
import cv2
import math
import numpy as np
from numpy import linalg
import logging
logger = logging.getLogger(__name__)

from . import utils


VERBOSE = 1


def align_images_ransac(*args, **kwargs):
    stitcher = AlignImagesRansac(*args, **kwargs)
    stitcher.start_image_stitching()




class AlignImagesRansac(object):

    def __init__(self, image_dir=None, key_frame=None, output_dir=None,
                 input_images=None, include_pattern=None,
                 score_max=0.7, score_min=0.1,
                 selection_score=0.4, selection_count_min=1,
                 output_filetype="PNG", output_fnfmt="{iteration}.{output_filetype}",
                 merge_lowres=True,
                 **kwargs):
        '''
        image_dir: 'directory' containing all images
        key_frame: 'dir/name.jpg' of the base image
        output_dir: 'directory' where to save output images
        optional:
           img_filter = 'JPG'; None->Take all files.

        score_min: minimum score required for any image to be appended.
        score_max: If an image has a score higher than this, it is immediately appended,
                   (without checking whether another image has an even higher score).
        selection_score: Repeat candidate search using images with a score above this.
        selection_count_min: If the selection list has less than this number of images,
                             then do not use selection; use full dir_list instead.
        '''

        self.output_filetype = output_filetype
        self.output_fnfmt = output_fnfmt
        self.output_dir = output_dir or os.path.join(".", "stitched-out")
        self.score_max = score_max
        self.score_min = score_min
        self.selection_score = selection_score
        self.selection_count_min = selection_count_min
        self.merge_lowres = merge_lowres
        self.show_images = False # Will show images as they are stitched together.
        self.eliminate_contours = False
        self.next_img_scores = {}
        self.key_frame_filepath = key_frame
        self.key_frame_basename = os.path.basename(key_frame) if key_frame else None
        self.image_dir = image_dir or "." if not input_images else None
        self.dir_list = input_images or []
        self.include_pattern = include_pattern
        self.kwargs = kwargs


    def get_image_dir_files(self):
        # Open the directory given in the arguments
        try:
            image_dir_filenames = os.listdir(self.image_dir)
        except OSError:
            print("Unable to open directory: %s" % self.image_dir)
            sys.exit(-1)
        # Filter input-dir filenames:
        if self.include_pattern:
            # remove all files that doen't end with .[image_filter]
            # TODO: Use glob or fnmatch-based filtering instead.
            if self.kwargs.get("include_pattern_fnmatch"):
                image_dir_filenames = [fn for fn in image_dir_filenames if fnmatch(fn, self.include_pattern)]
            else:
                image_dir_filenames = [fn for fn in image_dir_filenames if self.include_pattern in fn]
        if 'Thumbs.db' in self.dir_list: # always remove Thumbs.db (windows only)
            image_dir_filenames.remove('Thumbs.db')
        # Join to get filepaths:
        filelist = [os.path.join(self.image_dir, fn) for fn in image_dir_filenames]
        nbefore = len(filelist)
        filelist = [fpath for fpath in filelist if os.path.isfile(fpath)]
        logger.debug("Files before/after filtering actual files: %s/%s", nbefore, len(filelist))
        return filelist


    def start_image_stitching(self, key_frame=None):
        """
        Find input images in dir_list and start stitching process.
        """

        ## Assert that we have a good output dir:
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        elif not os.path.isdir(self.output_dir):
            raise RuntimeWarning("output_dir is not a directory: %s" % self.output_dir)


        if self.image_dir:
            image_dir_files = self.get_image_dir_files()
            logger.debug("image_dir_files: %s", image_dir_files)
            if not image_dir_files:
                print("No matching files was found in the image directory:", self.image_dir)
            self.dir_list += image_dir_files

        # Filter out anything that is not a file:
        self.dir_list = [fpath for fpath in self.dir_list if os.path.isfile(fpath)]
        logger.info("Processing %s files...", len(self.dir_list))
        if not self.dir_list:
            print("Input image file list is empty (%s) - aborting image stitching" % self.dir_list)
            return

        if key_frame is None:
            key_frame = self.key_frame_filepath
            if key_frame is None:
                # Sort images by size and use the largest one.
                self.dir_list.sort(key=os.path.getsize, reverse=True)
                key_frame = self.key_frame_filepath = self.dir_list.pop(0)
                self.key_frame_basename = os.path.basename(key_frame)
                print("Using key frame:", key_frame)
        print("Reading keyframe file:", key_frame)
        base_img_rgb = cv2.imread(key_frame)
        if base_img_rgb is None:
            raise IOError("%s doesn't exist" % key_frame)
        # utils.showImage(base_img_rgb, scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()

        # Remove keyframe image from dir_list:
        self.dir_list = [fpath for fpath in self.dir_list if fpath != key_frame]
        if VERBOSE:
            print("dir_list:", self.dir_list)


        ## I believe feature detector and matcher are both re-usable: ##

        # Use the SIFT feature detector
        self.detector = cv2.SIFT()

        # Parameters for nearest-neighbor matching
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})

        final_img = self.stitchImages(base_img_rgb, 0)
        return final_img


    def calculate_homography_stats(self, base_features, base_descs, next_img_path):
        # Read in the next image...
        print("Reading next_img_path:", next_img_path)
        next_img_rgb = cv2.imread(next_img_path)
        next_img = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        # Find points in the next frame
        next_features, next_descs = self.detector.detectAndCompute(next_img, None)
        matches = self.matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)
        matches_subset = utils.filter_matches(matches)
        distance = utils.imageDistance(matches_subset)
        average_point_distance = distance/len(matches_subset)
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
            print("\t Average Distance: ", average_point_distance)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

        stats = {'h': H,
                 'inliers': inlierRatio,
                 'dist': average_point_distance,
                 'path': next_img_path,
                 'rgb': next_img_rgb,
                 'img': next_img,
                 'feat': next_features,
                 'desc': next_descs,
                 'match': matches_subset
                }
        return stats


    def find_closest_image(self, base_img, candidates=None):
        # Find key points in base image for motion estimation
        base_features, base_descs = self.detector.detectAndCompute(base_img, None)

        if self.show_images:
            # Create new key point list
            # Ever heard of list comprehensions? No? OK.
            key_points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in base_features]
            utils.showImage(base_img, key_points, scale=(0.2, 0.2), timeout=0)
            cv2.destroyAllWindows()

        print("Iterating through next images...")

        closestImage = None

        # TODO: Thread this loop since each iteration is independent

        # Find the best next image from the remaining images
        if candidates is None:
            candidates = self.dir_list

        for next_img_path in candidates:

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

            if self.key_frame_basename in next_img_path:
                print("\t Skipping keyframe image %s..." % self.key_frame_basename)
                continue
            homography_stats = self.calculate_homography_stats(base_features, base_descs, next_img_path)
            self.next_img_scores[next_img_path] = homography_stats['inliers']

            # What is the best scoring criteria? inlierRatio or average_point_distance?
            if closestImage is None or homography_stats['inliers'] > closestImage['inliers']:
                closestImage = homography_stats
                if homography_stats['inliers'] > self.score_max:
                    print("%s is above score_max, combining immediately." % next_img_path)
                    break

        print("Closest Image: ", closestImage['path'])
        print("Closest Image Ratio: ", closestImage['inliers'])
        #self.dir_list = [x for x in self.dir_list if x != closestImage['path']]
        try:
            self.dir_list.remove(closestImage['path'])  # more expressive
            candidates.remove(closestImage['path'])
        except ValueError:
            pass
        # sort files by their score to increase chance they will break early next time
        self.dir_list.sort(key=lambda img_path: self.next_img_scores.get(img_path, 0))
        if closestImage['inliers'] < self.score_min:
            print("homography_stats['inliers'] < self.score_min, returning None:")
            print("%s < %s" % (closestImage['inliers'], self.score_min))
            return None
        return closestImage

    def combine_closest_image(self, base_img_rgb, closestImage, H_inv):
        """
        Does dependent combining of base_img and the closest image.
        """
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

        return final_img


    def stitchImages(self, base_img_rgb, iteration=0, selection=None):
        """
        Recursively stitch images in self.dir_list together, starting with base_img_rgb.
        Finds the closest image match of the (remaining) files in self.dir_list and
        combines that image with base_img_rgb.
        """
        if not self.dir_list:
            print("Dir list is exhausted - all done.")
            return base_img_rgb

        base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        if selection:
            print("Searching for suitable candidate among selection:", selection)
            closestImage = self.find_closest_image(base_img, candidates=selection)
        else:
            closestImage = None
        # There is a small posibility that find_closest_image does not find a suitable candidate
        # from the selection, in which case we should repeat the search for all images:
        if closestImage is None:
            closestImage = self.find_closest_image(base_img)

        if closestImage is None or closestImage['inliers'] < self.score_min:
            # No suitable candidates...
            # Uh... if we didn't find any candidates here, we're not gonna find it in any other rounds either.
            # Probably just abort and return?
            print("Closest image was:", closestImage['path'] if closestImage else closestImage)
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

        # http://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix
        scale_x = math.sqrt(H[0,0]**2 + H[0,1]**2)
        scale_y = math.sqrt(H[1,0]**2 + H[1,1]**2)

        if self.merge_lowres or H[0,0] > 1 or H[1,1] > 1 or scale_x > 1 or scale_y > 1:
            # Uh, even if the next image is low res, we might still want to
            # merge it if it expands the current canvas.
            # This probably gets too difficult to get right.
            final_img = self.combine_closest_image(base_img_rgb, closestImage, H_inv)

            # Write out the current round
            final_filename = self.output_fnfmt.format(iteration=iteration, round=iteration,
                                                      output_filetype=self.output_filetype)
            final_filepath = os.path.join(self.output_dir, final_filename)
            success = cv2.imwrite(final_filepath, final_img)
            print("(success)" if success else "(failed)", "Writing combined image to file", final_filepath)

        else:
            print("Closest image is lower res than base image and will not be merged. -", closestImage['path'])
            # Wait, if we are not changing the base image, the scores will be exactly the same.
            # So there should be a way to avoid the search and simply use the next-best candidate...
            final_img = base_img_rgb

        selection = [img_path for img_path in (selection or self.dir_list)
                     if self.next_img_scores.get(img_path, 0) > self.selection_score]
        if not selection or len(selection) < self.selection_count_min:
            selection = None

        return self.stitchImages(final_img, iteration+1, selection=selection)






 # ----------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: %s <image_dir> <key_frame> <output>" % sys.argv[0])
        sys.exit(-1)
    print("sys.argv:", sys.argv[1:])
    #AlignImagesRansac(*sys.argv[1:])
    align_images_ransac(*sys.argv[1:])
