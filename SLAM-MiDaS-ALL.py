import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import open3d as o3d

num_frames = 2

server_address = 'http://172.20.10.4:8080/shot.jpg'

camera_matrix = np.load("camera_matrix.npy")

# Load distortion coefficients
dist_coeffs = np.load("dist_coeffs.npy")

model_type = "MiDaS_small"
# model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
print("Model loaded")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def feature_extraction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)
    kp, desc = orb.compute(gray, kp)

    return kp, desc


def feature_matching(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)

    return matches


def feature_extraction_sift(image):
    sift = cv2.SIFT_create(nfeatures=2000)
    kp, desc = sift.detectAndCompute(image, None)

    return kp, desc


def feature_matching(image1, image2, top_k=50):
    flann = cv2.FlannBasedMatcher()

    kp1, desc1 = feature_extraction_sift(image1)
    kp2, desc2 = feature_extraction_sift(image2)
    matches = flann.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:top_k]

    matched_image = cv2.drawMatches(
        image1, kp1,
        image2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Matches", matched_image)

    return matches, kp1, kp2, desc1, desc2


def filter_matches(matches):
    good_matches = []
    for match in matches:
        if len(match) == 2:
            good_matches.append(match)
    return good_matches


depth_frame = ...

R = np.eye(3)
t = np.zeros((3, 1))
proj_matrix1 = camera_matrix @ np.hstack((R, t))

proj_matrix2 = ...

height = 1024
width = 1024

point_cloud = np.empty((0, 3))
colors = []

pose = np.eye(4)
trajectory = np.zeros((600, 800, 3), dtype=np.uint8)
prev_frame = None

for frame_idx in range(num_frames):
    print("Processing frame: ", frame_idx)

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 represents the default camera
    ret, frame = cap.read()
    cap.release()

    input_batch = transform(frame).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    output = (output - output.min()) * (255 / (output.max() - output.min()))
    depth_frame = output.astype(np.uint8)

    if prev_frame is not None:
        matches, kp1, kp2, desc1, desc2 = feature_matching(
            frame, prev_frame)
        good_matches = []
        good_matches = [match for match in matches if match.distance < 10]

        if len(good_matches) > 15:
            pts1 = np.float32(
                [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts2 = np.float32(
                [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 3.0)

            _, R, t, _ = cv2.recoverPose(
                F, pts1, pts2, cameraMatrix=camera_matrix)

            proj_matrix2 = camera_matrix @ np.hstack((R, t.reshape(3, 1)))
            pose[:3, :3] = R
            pose[:3, 3] = t.flatten()

            # Extract rotation angles from the pose matrix
            pitch = np.arctan2(pose[2, 1], pose[2, 2])
            yaw = np.arctan2(-pose[2, 0],
                             np.sqrt(pose[2, 1]**2 + pose[2, 2]**2))
            roll = np.arctan2(pose[1, 0], pose[0, 0])

            for i, match in enumerate(good_matches):
                pt1 = kp1[match.queryIdx].pt
                pt2 = kp2[match.trainIdx].pt
                pt1 = (int(pt1[0]), int(pt1[1]))
                pt2 = (int(pt2[0]), int(pt2[1]))
                depth = depth_frame[pt2[1], pt2[0]]

                if depth > 0:
                    point_3d = cv2.triangulatePoints(
                        proj_matrix1, proj_matrix2, pt1, pt2)
                    point_3d /= point_3d[3]
                    point_cloud = np.array(point_cloud)
                    point_cloud = np.concatenate(
                        (point_cloud, point_3d.T[:, :3]), axis=0)

                    colors.append(frame[pt2[1], pt2[0]])

            # Draw the coordinate axes
            axis_length = 0.1 * min(height, width)
            imgpts, _ = cv2.projectPoints(np.float32([[0, 0, 0], [axis_length, 0, 0], [
                0, axis_length, 0], [0, 0, axis_length]]), R, t, np.eye(3), None)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            cv2.line(trajectory, tuple(imgpts[0]),
                     tuple(imgpts[1]), (0, 0, 255), 3)
            cv2.line(trajectory, tuple(imgpts[0]),
                     tuple(imgpts[2]), (0, 255, 0), 3)
            cv2.line(trajectory, tuple(imgpts[0]),
                     tuple(imgpts[3]), (255, 0, 0), 3)

    prev_frame = frame


distance_threshold = 500.0

# Convert the point cloud to an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Compute the distances of each point from the origin
distances = np.linalg.norm(point_cloud, axis=1)

filtered_indices = np.where(distances <= distance_threshold)[0]
filtered_pcd = pcd.select_by_index(filtered_indices)

o3d.visualization.draw_geometries([filtered_pcd])

cv2.imshow("Depth", depth_frame)

cv2.imshow("Trajectory", trajectory)

cv2.waitKey(0)
