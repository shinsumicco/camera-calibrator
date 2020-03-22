#!/usr/bin/env python
import re
import cv2
import argparse
import numpy as np
from pathlib import Path


GRID_WIDTH = 29.8  # mm
GRID_COLS = 13
GRID_ROWS = 6
GRID_NUM = GRID_COLS * GRID_ROWS


def execute(dn_rgb_img: Path, dn_rgb_txt: Path, dn_calib: Path):
    if not dn_rgb_txt.exists() or not dn_rgb_txt.is_dir():
        dn_rgb_txt.mkdir()

    # load images
    fp_rgb_img_list = [p for p in dn_rgb_img.glob("*") if re.search(".*(jpg|JPG|png|PNG)", str(p))]
    fp_rgb_img_list = sorted(fp_rgb_img_list)

    # prepare 3D points
    obj_coords = np.zeros((1, GRID_COLS * GRID_ROWS, 3), np.float32)
    obj_coords[0, :, :2] = np.mgrid[0:GRID_COLS, 0:GRID_ROWS].T.reshape(-1, 2) * GRID_WIDTH

    objpoints = []
    imgpoints = []

    print("----- Circle Grid Detection -----")
    for fp_rgb_img in fp_rgb_img_list:
        print("{}".format(fp_rgb_img))
        rgb_img = cv2.imread(str(fp_rgb_img), cv2.IMREAD_GRAYSCALE)
        rgb_img = rgb_img.astype(np.uint8)

        # detect circle grid centers
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10 ** 2
        params.maxArea = 100 ** 2
        params.filterByColor = True
        params.minThreshold = 0
        params.maxThreshold = 192
        params.filterByCircularity = True
        params.filterByInertia = False
        params.filterByConvexity = True
        detector = cv2.SimpleBlobDetector_create(params)
        detect_flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
        is_found, centers = cv2.findCirclesGrid(rgb_img, (GRID_COLS, GRID_ROWS), detect_flags, blobDetector=detector)

        # check if centers are valid or not
        if not is_found or centers is None or len(centers) != GRID_NUM:
            print("- FAILED")
            continue

        # convert to color image format
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
        cv2.drawChessboardCorners(rgb_img, (GRID_COLS, GRID_ROWS), centers, True)
        center_coords = []
        for i, corner in enumerate(centers):
            cv2.putText(rgb_img, str(i), (int(corner[0][0]) + 2, int(corner[0][1]) - 2),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            center_coords.append((corner[0][0], corner[0][1]))

        objpoints.append(obj_coords)
        imgpoints.append(centers)

        cv2.imwrite(str(dn_rgb_txt / fp_rgb_img.name), rgb_img)
        np.savetxt(str(dn_rgb_txt / (fp_rgb_img.stem + ".txt")), np.array(center_coords))

    # convert to grayscale image format
    calib_flags = cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_TANGENT_DIST
    criteria_flags = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-12)

    # dummy input (these initial values won't be used because cv2.CALIB_USE_INTRINSIC_GUESS is not specified)
    camera_matrix = np.zeros((3, 3))
    dist_params = np.zeros((5, 1))
    rmse, camera_matrix, dist_params, rvecs, tvecs \
        = cv2.calibrateCamera(objpoints, imgpoints, (rgb_img.shape[1], rgb_img.shape[0]),
                              camera_matrix, dist_params, flags=calib_flags, criteria=criteria_flags)
    print("")

    print("----- Calibration Results -----")
    print("RMSE:")
    print(rmse)
    print("K:")
    print(camera_matrix)
    print("D:")
    print(dist_params.T)
    print("")

    print("----- Undistortion of Calibration Images -----")

    if not dn_calib.exists() or not dn_calib.is_dir():
        dn_calib.mkdir()

    np.savetxt(dn_calib / "camera_matrix.txt", camera_matrix)
    np.savetxt(dn_calib / "dist_params.txt", dist_params)

    # create a new optimal camera matrix
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_params,
                                                         (rgb_img.shape[1], rgb_img.shape[0]), 1.0)
    remap = cv2.initUndistortRectifyMap(camera_matrix, dist_params, np.eye(3), new_camera_matrix,
                                        (rgb_img.shape[1], rgb_img.shape[0]), cv2.CV_32FC1)
    for fp_rgb_img in fp_rgb_img_list:
        rgb_img = cv2.imread(str(fp_rgb_img), cv2.IMREAD_COLOR)
        rgb_img = rgb_img.astype(np.uint8)
        rgb_img = cv2.remap(rgb_img, remap[0], remap[1], cv2.INTER_AREA)
        fp_rgb_img_undist = dn_rgb_txt / ("undist_" + fp_rgb_img.name)
        print(f"- {fp_rgb_img_undist}")
        cv2.imwrite(str(fp_rgb_img_undist), rgb_img)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration script for a perspective camera")
    parser.add_argument("dp_rgb_img", type=Path, help="directory of RGB images")
    parser.add_argument("dp_rgb_txt", type=Path, help="directory of circle detection output")
    parser.add_argument("dp_calib", type=Path, help="directory of calibration output")
    args = parser.parse_args()

    execute(args.dp_rgb_img, args.dp_rgb_txt, args.dp_calib)
