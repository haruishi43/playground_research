#!/usr/bin/env python3

"""This module converts fisheye image into panoramic image.
"""

import math

import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


from fisheye_unwarp.utils import math as math_util

PI = math.pi


def equ_to_vector(x, y, r=1.0):
    """This function converts a coordinate in 2D Equirectangular projection to its 3D point.
    It assume the projection is on a sphere (which it converts back to).

    Args:
        x: Horizontal coordinate
        y: Vertival coordinate
        r: The radius of the sphere

    Return:
        Px: x of 3D point
        Py: y of 3D point
        Pz: z of 3D point
    """
    lon = x * PI
    lat = y * PI / 2.0

    Px = r * math.cos(lat) * math.cos(lon)
    Py = r * math.cos(lat) * math.sin(lon)
    Pz = r * math.sin(lat)
    return Px, Py, Pz


def vector_to_r_theta(px, py, pz, aperture):
    """This function converts a coordinate on a sphere,
    to r, theta in projection. What we have is an axis, y axis
    that is where our lens is pointing to, and r defines the
    angle of y axis to the point p which is defined by px, py, pz.
    Theta is the rotation of the point p on y axis.
    Note: P = (Px, Py, Pz), |P| = 1.

    Args:
        px: x of 3D point
        py: y of 3D point
        pz: z of 3D point

    Return:
        r: r, Angle of PO and y axis
        theta: Rotation of point p on y axis.
    """
    r = 2 * math.atan2(math.sqrt(px * px + pz * pz), py) / aperture
    theta = math.atan2(pz, px)
    return r, theta


def calculate_error(r, t, pts_a, pts_b):
    """This function calculates error by measuring the distance
    between corresponding r(p_a) + t and p_b

    Args:
        r: 3x3 np rotation matrix
        t: 3x1 np translation array
        pts_a: nx3 3D points
        pts_b: nx3 3D points

    Return:
        diff: Average difference of transformed a and b points
    """
    diff = 0.0
    for p_a, p_b in zip(pts_a, pts_b):
        diff += np.linalg.norm(r.dot(p_a) + t - p_b)
    return diff / len(p_a)


def vector_to_r_theta_with_delta_y(px, py, pz, aperture, dy, rotation_matrix):
    """This function translates 3d point p on sphere to r, theta.

    Args:
        px: x of p
        py: y of p
        pz: z of p
        aperture: Field of view of the lens.
        dy: 3D vector that shows where is the optical center of the lens.
        rotation_matrix: The rotation of the lens,
            lens points to y axis when it is an eye matrix.

    Returns:
        r: Angle between dy, y axis, and point p
        theta: Rotation on y axis.
    """
    vec = np.array([px, py, pz]) - dy
    vec = vec / np.linalg.norm(vec)
    px, py, pz = rotation_matrix.dot(vec)

    r = 2 * math.atan2(math.sqrt(px * px + pz * pz), py) / aperture
    theta = math.atan2(pz, px)
    return r, theta


def vector_to_fisheye(px, py, pz, aperture, det_y, rotation_matrix=np.eye(3)):
    """This function takes a 3D point on unit sephere,
    and map it to 2D fish eye coordinate.

    Args:
        px: x of 3D point
        py: y of 3D point
        pz: z of 3D point
        aperture: Field of view of the image in radius.
        det_y: The difference between lens principal point
            and origin point(0,0,0), we only allow 1D adjust.

    Return:
        x: x of 2D point on normalized plane
        y: y of 2D point on normalized plane
    """
    r, theta = vector_to_r_theta_with_delta_y(
        px, py, pz, aperture, det_y, rotation_matrix
    )

    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y


def generate_map(
    input_image_size,
    output_image_size,
    aperture,
    rotation_matrix,
    o_center_position,
):
    """This function generates a map for openCV's remap function.

    Args:
        input_image_size: Tuple of the input image size, (y,x)
        output_image_size: Tuple of the output image size, (y,x)
        aperture: Field of view of the image in radius.
        rotation_matrix: Not used for now
        o_center_position: 3D vector, the position of the optical center.

    Return:
        Matrix: A matrix same size as output_image.
    """
    print("input_image_size", input_image_size)
    print("output_image_size", output_image_size)
    if output_image_size[0] * 2 != output_image_size[1]:
        print("error output_image_size")
    else:
        image_map = np.zeros(
            (output_image_size[0], output_image_size[1], 2), dtype=np.float32
        )
        for x in range(output_image_size[1]):
            for y in range(output_image_size[0]):
                normal_x = math_util.lerp(x, 0, output_image_size[1] - 1, -1, 1)
                normal_y = -math_util.lerp(
                    y, 0, output_image_size[0] - 1, -1, 1
                )  # invert y value

                px, py, pz = equ_to_vector(normal_x, normal_y, r=1000.0)
                normal_fish_x, normal_fish_y = vector_to_fisheye(
                    px,
                    py,
                    pz,
                    aperture,
                    o_center_position,
                    rotation_matrix,
                )

                fish_x = math_util.lerp(
                    normal_fish_x, -1, 1, 0, input_image_size[1] - 1
                )
                fish_y = math_util.lerp(
                    normal_fish_y, -1, 1, 0, input_image_size[0] - 1
                )

                image_map[y, x] = fish_x, fish_y
        return image_map


def convert_fisheye_equ(
    input_image, output_image_size, aperture, rotation_matrix, o_center_position
):
    """This function convert an equidistant projected fisheye lens
    into an equirectangular projection image.

    Args:
        input_image: input image should be np matrix
        output_image_size: Tuple of the output image size, (y,x)
        aperture: Fild of view of the image in radius.
        rotation_matrix: Not used for now

    Return:
        matrix: unwarped image.
    """
    image = np.zeros(
        (output_image_size[1], output_image_size[0], 3),
        dtype=np.uint8,
    )
    image_map = generate_map(
        input_image.shape,
        image.shape,
        aperture,
        rotation_matrix,
        o_center_position,
    )

    cv2.remap(
        src=input_image,
        dst=image,
        map1=image_map,
        map2=None,
        interpolation=cv2.INTER_CUBIC,
    )
    # cv2.imwrite(output_image_name,image)
    # cv2.imshow("aaa",image)
    # c = cv2.waitKey(0)
    return image
