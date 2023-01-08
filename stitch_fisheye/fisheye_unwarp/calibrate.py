#!/usr/bin/env python3

import math

import cv2

import numpy as np

from fisheye_unwarp import find_checkerboard
from fisheye_unwarp.utils import math as math_util


def fisheye_coord_to_phi_theta(fish_coord, aperture):
    """This function translates fisheye coordinate to phi, theta.
    Please see: http://paulbourke.net/dome/dualfish2sphere/

    Args:
        fish_coord: 3(?)x1 vector. The values should be normalized,
            where x = [-1,1], y = [-1,1]
        aperture: The Field of view in radius.

    Returns:
        phi: phi in radius.
        theta: Theta in radius.
    """
    phi = np.linalg.norm(fish_coord) * aperture / 2.0
    theta = math.atan2(fish_coord[1], fish_coord[0])
    return phi, theta


def phi_theta_to_pxyz(
    phi,
    theta,
    o_center_position=np.array([0.0, 0.0, 0.0]),
    r=1.0,
    rotation_matrix=np.eye(3),
):
    """This function translates phi, theta to projected 3d sphere,
    with respect of optical center, radius, and rotation.

    Args:
        phi: phi in radian.
        theta: Theta in radian.
        o_center_position: A 3d vector shows where is the optical center.
        r: The radius of the projected sphere, |p| = r
        rotation_matrix: The rotation of the optical axis,
            it is originally pointed to y axis.

    Returns:
        A 3d vector, that is on sphere.
    """
    vec = np.array(
        [
            math.sin(phi) * math.cos(theta),
            math.cos(phi),
            math.sin(theta) * math.sin(phi),
        ]
    )
    vec = rotation_matrix.dot(vec)
    o_center_length = np.linalg.norm(o_center_position)
    x1, _ = math_util.solve_quadratic(
        1.0,
        2.0 * np.inner(vec, o_center_position),
        o_center_length * o_center_length - r * r,
    )
    return x1 * vec + o_center_position


def reproject_coords(coords, fov, r, o_center_position, rotation_matrix):
    result = np.zeros((len(coords), 3))
    for i, fish_coord in enumerate(coords):
        phi, theta = fisheye_coord_to_phi_theta(fish_coord, fov)
        # print(phi,theta)
        result[i] = phi_theta_to_pxyz(
            phi,
            theta,
            o_center_position,
            r,
            rotation_matrix,
        )
    return result


def pxyz_to_equ(p_coord):
    """This function translates 3D point p on sphere to
    Equirectangular projection 2d point.

    Args:
        p_coord:  A 3d vector, that is on sphere.

    Returns:
        lon: longitude of the point p. Range from[-1,1]
        lat: latitude of the point p. Range from[-1,1]
    """
    lon = math.atan2(p_coord[1], p_coord[0])
    lat = math.atan2(p_coord[2], np.linalg.norm(p_coord[0:2]))
    return lon / math.pi, 2.0 * lat / math.pi


def calculate_error_2D(points_a, points_b, r):
    difference = 0.0
    points_a /= r
    points_b /= r
    i = 0
    for p_a, p_b in zip(points_a, points_b):
        coord_a = np.array(pxyz_to_equ(p_a))
        coord_b = np.array(pxyz_to_equ(p_b))
        i += 1
        difference += np.linalg.norm(coord_a - coord_b)

    return difference / len(points_b)


def print_points_2d(points_a, points_b, r):
    img = np.zeros((1024, 2048, 3), dtype=np.uint8)
    points_a /= r
    points_b /= r
    for p_a in points_a:
        coord_a = np.array(pxyz_to_equ(p_a))
        coord_a_x = int(math_util.lerp(coord_a[0], -1, 1, 0, 2048))
        coord_a_y = int(math_util.lerp(coord_a[1], -1, 1, 0, 1024))
        img[coord_a_y, coord_a_x, :] = [255, 0, 0]

    for p_b in points_b:
        coord_b = np.array(pxyz_to_equ(p_b))
        coord_b_x = int(math_util.lerp(coord_b[0], -1, 1, 0, 2048))
        coord_b_y = int(math_util.lerp(coord_b[1], -1, 1, 0, 1024))
        img[coord_b_y, coord_b_x, :] = [0, 255, 0]
    cv2.imwrite("template.png", img)


def dual_fisheye_calibrate(
    image_l,
    image_r,
    fov_l,
    fov_r,
    points_a=None,
    points_b=None,
):

    # for both image

    # Calibration Markers

    # left
    #     2     1
    #
    #  3
    #
    #
    #  4
    #
    #     5     6

    # right
    #     1     2
    #
    #              3
    #
    #
    #              4
    #
    #     6     5

    image_boards = np.array(
        [
            [
                [1 / 2.0, 0.0, 1 / 2.0, 1 / 2.0],
                [1 / 4.0, 0.0, 1 / 4.0, 1 / 4.0],
                [0.0, 1 / 4.5, 1 / 4.0, 1 / 4.0],
                [0.0, 1 / 2.0, 1 / 3.0, 1 / 3.0],
                [1 / 4.0, 3 / 4.0, 1 / 4.0, 1 / 4.0],
                [1 / 2.0, 3 / 4.0, 1 / 4.0, 1 / 4.0],
            ],
            [
                [1 / 4.0, 0.0, 1 / 4.0, 1 / 4.0],
                [1 / 2.0, 0.0, 1 / 4.0, 1 / 4.0],
                [3 / 4.0, 1 / 4.5, 1 / 4.0, 1 / 4.0],
                [3 / 4.0, 1 / 2.0, 1 / 3.0, 1 / 3.0],
                [1 / 2.0, 3 / 4.0, 1 / 4.0, 1 / 4.0],
                [1 / 4.5, 3 / 4.0, 1 / 4.0, 1 / 4.0],
            ],
        ]
    )
    all_points = []
    # ite = 100

    if points_a is not None and points_b is not None:
        all_points = [points_a, points_b]
    else:
        # 1. find points
        for image, boards in zip([image_l, image_r], image_boards):
            h, w = image.shape[:2]
            length = np.array(image.shape[:2])

            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            board_points = np.zeros((0, 2))
            for board in boards:
                mask = np.array(
                    [board[0] * w, board[1] * h, board[2] * w, board[3] * h],
                    dtype=np.int32,
                )
                point = find_checkerboard.find_checkerboard(mask, gray_img)
                point = np.squeeze(point, axis=1)
                print(
                    board_points.shape, point.shape
                )  # see if it appends the corners
                board_points = np.vstack((board_points, point))
            board_points /= length / 2.0
            # print(np.tile([1.0,1.0],(len(board_points),2)))
            board_points -= np.tile([1.0, 1.0], (len(board_points), 1))
            all_points.append(board_points)

    # 2. unwarp and project to global pxyz
    def find_r_t(
        fov_l,
        fov_r,
        image_center_l=np.array([0.0, 0.0]),
        image_center_r=np.array([0.0, 0.0]),
    ):
        rotation_matrix_l = np.eye(3)
        # math_util.rotation_matrix_z(math.pi/2.0)
        rotation_matrix_r = np.eye(3)
        r = 1.0  # 1000.0  # 1 meter?
        det_y_l = np.array([0.0, 0.0, 0.0])
        det_y_r = np.array([0.0, 0.0, 0.0])
        # ite = 50
        eps = 1e-10
        old_error = float("inf")
        error = 1000.0

        while old_error - error > eps:
            l_reproject_pts = reproject_coords(
                all_points[0] + image_center_l,
                fov_l,
                r,
                det_y_l,
                rotation_matrix_l,
            )
            r_reproject_pts = reproject_coords(
                all_points[1] + image_center_r,
                fov_r,
                r,
                det_y_r,
                rotation_matrix_r,
            )

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(l_reproject_pts[:,0], l_reproject_pts[:,1],l_reproject_pts[:,2],c='r')
            # ax.scatter(r_reproject_pts[:,0], r_reproject_pts[:,1], r_reproject_pts[:,2],c='b')
            # plt.show()
            # error =  calculate_error(np.eye(3), np.zeros((1,3)),l_reproject_pts,r_reproject_pts)
            old_error = error
            error = calculate_error_2D(
                l_reproject_pts.copy(), r_reproject_pts.copy(), 1000.0
            )

            # 3. find translation + rotation for points
            new_rotation, t = math_util.find_rigid_transformation(
                np.mat(r_reproject_pts),
                np.mat(l_reproject_pts),
            )

            # 4. calculate error
            # print(rotation_matrix_l)

            det_y_l -= np.array(t)[:, 0] / 2.0
            det_y_r += np.array(t)[:, 0] / 2.0

            np.matmul(new_rotation, rotation_matrix_r, rotation_matrix_r)

            # print(calculate_error(new_rotation, t,l_reproject_pts,r_reproject_pts))
        # print_points_2d(l_reproject_pts.copy(), r_reproject_pts.copy(), 1000.0)
        return error, rotation_matrix_l.T, det_y_l, rotation_matrix_r.T, det_y_r

    min_value = float("inf")
    min_result = None
    min_fov = None

    for d_fov_l in range(-0, 1, 1):
        for d_fov_r in range(-0, 1, 1):
            # for d_img_center_l_x in range(-3, 3, 1):
            #     for d_img_center_l_y in range(-3, 3, 1):
            #         for d_img_center_r_x in range(-3, 3, 1):
            #             for d_img_center_r_y in range(-3, 3, 1):
            try:
                result = find_r_t(
                    fov_l + d_fov_l / 180.0 * math.pi,
                    fov_r + d_fov_r / 180.0 * math.pi,
                )  # np.array([d_img_center_l_x,d_img_center_l_y])/3000.0,np.array([d_img_center_r_x,d_img_center_r_y])/3000.0
            except Exception:
                raise
            if result[0] < min_value:
                min_value = result[0]
                min_result = result
                min_fov = (
                    fov_l + d_fov_l / 180.0 * math.pi,
                    fov_r + d_fov_r / 180.0 * math.pi,
                )

    return min_result, min_fov
