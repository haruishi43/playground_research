#!/usr/bin/env python3

import cv2

import numpy as np

i = 0


def find_checkerboard(mask_rect, image, pattern_size=(6, 3)):
    """This function finds points on checkerbord.

    Args:
        mask_rect: (x,y,w,h), Part of the image that is used to find checkerboard,
        x,y is the top left start point. w,h is the width and height of the image.
        image: Gray scale image
        pattern_size: The size of checker board.

    Return:
        corners: The founded checker board.

    Note, if mask is set than the corners is compensated for masking.
    """
    global i
    img_h, img_w = image.shape[:2]

    subImage = image[
        mask_rect[1] : min(mask_rect[1] + mask_rect[3], img_h),
        mask_rect[0] : min(mask_rect[0] + mask_rect[2], img_w),
    ]
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

    # find corners
    found, corners = cv2.findChessboardCorners(subImage, pattern_size)

    # if found refine by subpix
    # cv2.imshow("aaa", subImage)
    # cv2.waitKey(3000)
    if found:
        # term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        # cv2.cornerSubPix(subImage, corners, (5, 5), (-1, -1), term)
        # vis = cv2.cvtColor(subImage, cv2.COLOR_GRAY2BGR)

        # cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        # cv2.imshow("aaa", vis)
        # cv2.imwrite("checkerboard_{0}.png".format(i), vis)
        i += 1

        corners[:, :, 0:2] += mask_rect[:2]
        return corners
    else:
        return None
