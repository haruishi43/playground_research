#!/usr/bin/env python3

import math

import cv2

import numpy as np

from fisheye_unwarp.unwarp import convert_fisheye_equ
from fisheye_unwarp.utils.math import rotation_matrix_z


# def test_rotation():
#     for i in range(-8, 8):
#         r = math.pi / 4.0 * i
#         r_m = rotation_matrix_z(r)
#         print(r_m)
#         input_image_l = cv2.imread("tests/_dataset/fisheye_l.jpg")
#         output_image = convert_fisheye_equ(
#             input_image_l,
#             (1024, 512),
#             200.0 / 180 * math.pi,  # FOV of 200 deg?
#             r_m,  # TODO: seems like horizontal rotation is not enough (yaw?)
#             np.array([0.0, 0.0, 0.0]),
#         )
#         cv2.imwrite("tests/_results/rotation_test_{}.jpg".format(i * 45), output_image)
