import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# IP camera stream URL
server_address = 'http://172.20.10.4:8080/shot.jpg'

# Load camera matrix
camera_matrix = np.load("camera_matrix.npy")

# Load distortion coefficients
dist_coeffs = np.load("dist_coeffs.npy")

model_type = "DPT_Large"
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

while True:
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    capture.release()

    img = frame

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    output = (output - output.min()) * (255 / (output.max() - output.min()))
    output = output.astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Output", output)

    output = output_bgr
    depth_map = (output - output.min()) / (output.max() - output.min())
    depth_map_float32 = depth_map.astype(np.float32)

    depth_image = o3d.geometry.Image(depth_map_float32)
    color_image = o3d.geometry.Image(frame)

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.intrinsic_matrix = camera_matrix

    # Create a point cloud from the depth and color images using the intrinsic parameters
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics)

    o3d.visualization.draw_geometries([point_cloud])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
