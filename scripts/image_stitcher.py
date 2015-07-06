#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##    Copyright 2015 Rasmus Scholer Sorensen, rasmusscholer@gmail.com
##

"""

Script for boot-strapping and executing image stitcher from command line.

"""

from __future__ import print_function, division, absolute_import
import os
import sys
import glob
import argparse
import yaml
import logging
logger = logging.getLogger(__name__)


try:
    from stitcher.align_images_ransac import AlignImagesRansac
except ImportError:
    # Add lib dir to system path:
    LIBDIRPATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    logger.debug("Appending sys.path with LIBDIRPATH: %s", LIBDIRPATH)
    sys.path.insert(0, LIBDIRPATH)
    from stitcher.align_images_ransac import AlignImagesRansac




def parse_args(argv=None):
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Image Stitcher - stitching images together since 2015.")
    parser.add_argument("--verbose", "-v", action="count", help="Increase verbosity.")
    parser.add_argument("--testing", action="store_true", help="Run app in simple test mode.")
    parser.add_argument("--loglevel", default=logging.INFO, help="Set logging output threshold level.")
    parser.add_argument("--logformat",
                        default="%(asctime)s %(levelname)-5s %(name)12s:%(lineno)-4s%(funcName)16s() %(message)s",
                        help="Set logging output format.")

    parser.add_argument("--no-merge-lowres", action="store_false", dest="merge_lowres",
                        help="Do not merge low-res images.")
    parser.add_argument("--merge-lowres", action="store_true",
                        help="Use all images, even images with lower resolution than the base image.")
    parser.add_argument("--highres-on-top", action="store_true",
                        help="Place the higher-res image on top. Default is to place the next/closest image on top.")
    parser.add_argument("--blendmode", default="lighten",
                        help="Image blending mode (think Photoshop).")

    parser.add_argument("--config", "-c",
                        help="Instead of providing a bunch of command line arguments at the command line, "
                        "you can read arguments from a dictionary stored as a yaml file).")

    parser.add_argument("--input-images", "-i", nargs="*",
                        help="Load images from this directory. Default is current directory.")
    parser.add_argument("--image-dir", "-d",
                        help="Load images from this directory. "\
                             "Default is current directory, unless input-images is provided.")
    parser.add_argument("--include-pattern",
                        help="Only include files matching this/these pattern(s).")
    parser.add_argument("--include-pattern-fnmatch", action="store_true",
                        help="Use 'fnmatch'-style matching with wild-card operators instead of simple string search.")

    parser.add_argument("--output-dir", "-o", default="stitched-output",
                        help="Output the stitched/combined images to this directory.")
    parser.add_argument("--output-fnfmt", default="{round}.{output_filetype}",
                        help="Filename format string for the stitched/combined image.")
    parser.add_argument("--output-filetype", default="PNG",
                        help="File format of the stitched/combined image. Relies on output-fnfmt.")

    parser.add_argument("--score-min", default=0.1,
                        help="Only combine/stitch on images with a matching score higher than this.")
    parser.add_argument("--score-max", default=0.6,
                        help="If an image has a score higher than this, "
                        "immediately stitch it onto the base image "
                        "(without testing to see if other images has an even higher score.")
    parser.add_argument("--score-selection", default=0.4,
                        help="Select images that initially score higher than this "
                        "and try to run only using these images first, then continue with the rest. "
                        "If you have a large canvas with lots of distant/unrelated images, "
                        "this can significantly reduce the computation time required to complete the "
                        "full stitching process.")

    parser.add_argument("--key-frame", help="Start stitching using this image. Defaults to the largest image.")


    return parser, parser.parse_args(argv)




def process_args(argns):
    """
    Process command line args and return a dict with args.

    If argns is given, this is used for processing.
    If argns is not given (or None), parse_args() is called
    in order to obtain a Namespace for the command line arguments.

    Will expand the entry "basedirs" using glob matching, and print a
    warning if a pattern does not match any files at all.

    If argns (given or obtained) contains a "config" attribute,
    this is interpreted as being the filename of a config file (in yaml format),
    which is loaded and merged with the args.

    Returns a dict.
    """
    args = argns.__dict__.copy()

    # Load config with parameters:
    if args.get("config"):
        with open(args["config"]) as fp:
            cfg = yaml.load(fp)
        args.update(cfg)

    if args.get("loglevel"):
        try:
            args["loglevel"] = int(args.get("loglevel"))
        except ValueError:
            args["loglevel"] = getattr(logging, args["loglevel"])

    # On Windows we have to expand glob patterns manually:
    # Also warn user if a pattern does not match any files
    if args.get('input_images'):
        file_pattern_matches = [(pattern, glob.glob(os.path.expanduser(pattern))) for pattern in args['input_images']]
        for pattern in (pattern for pattern, res in file_pattern_matches if len(res) == 0):
            print("WARNING: File/pattern '%s' does not match any files." % pattern)
        args['input_images'] = [fname for pattern, res in file_pattern_matches for fname in res]

    return args

def init_logging(args):
    """ Initialize logging based on args parameters. """
    default_fmt = "%(asctime)s %(levelname)-5s %(name)12s:%(lineno)-4s%(funcName)16s() %(message)s"
    logging.basicConfig(level=args.get("loglevel", 30),
                        format=args.get("logformat", default_fmt))




def main(argv=None):
    """ Main driver to parse arguments and start stitching. """
    _, argns = parse_args(argv)
    args = process_args(argns)
    init_logging(args)
    logger.debug("args: %s", args)

    stitcher = AlignImagesRansac(**args)
    stitcher.start_image_stitching()




if __name__ == '__main__':
    main()
