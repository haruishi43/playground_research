#!/usr/bin/env python3

import math

import cv2

from fisheye_unwarp.calibrate import dual_fisheye_calibrate
from fisheye_unwarp.unwarp import convert_fisheye_equ


if __name__ == "__main__":

    output_width = 1024

    input_image_r = cv2.imread("dataset/fisheye_r.jpg")
    input_image_l = cv2.imread("dataset/fisheye_l.jpg")

    result, fov = dual_fisheye_calibrate(
        input_image_l,
        input_image_r,
        200 / 180.0 * math.pi,
        200 / 180.0 * math.pi,
    )
    print(fov[0] / math.pi * 180, fov[1] / math.pi * 180)
    error, r_l, t_l, r_r, t_r = result
    print("Reproject Error:", error)
    print("Left image rotation matrix:")
    print(r_l)
    print("Left image Translation:")
    print(t_l)
    print("Right image rotation matrix:")
    print(r_r)
    print("Right image Translation")
    print(t_r)

    # e = np.array(math_util.rotation_matrix_decompose(r_r))
    # print(e / math.pi*180)

    output_image_l = convert_fisheye_equ(
        input_image_l,
        (output_width, output_width // 2),
        fov[0],
        r_l,
        t_l,
    )
    output_image_r = convert_fisheye_equ(
        input_image_r,
        (output_width, output_width // 2),
        fov[1],
        r_r,
        t_r,
    )
    cv2.imwrite(f"results/out_l_{output_width}.jpg", output_image_l)
    cv2.imwrite(f"results/out_r_{output_width}.jpg", output_image_r)
