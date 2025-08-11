import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def Change_of_basis(x,y, M,w1,h1,w2,h2):
# Tính ma trận nghịch đảo của M
    x_prime = x*w2
    y_prime = y*h2
    #M_inv = np.linalg.inv(M)

    # Chuyển tọa độ (x', y') về tọa độ trên ảnh gốc
    point_transformed = np.array([x_prime, y_prime, 1])  # Tọa độ chấm đen trên ảnh xử lý
    point_original = np.dot(M, point_transformed)   # Phép biến đổi ngược

    # Chuẩn hóa tọa độ
    x_original = point_original[0] / point_original[2]/w1
    y_original = point_original[1] / point_original[2]/h1

    return x_original, y_original

def change_coor(template_path, image_path, list_coor):

    # Step 1: Đọc ảnh
    img1 = Image.open(template_path)
    img2 = Image.open(image_path)

    w1, h1 = img1.size
    w2, h2 = img2.size

    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Step 2: Chuyển ảnh về grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Phát hiện keypoint và tính toán mô tả đặc trưng (descriptor) bằng ORB
    orb = cv2.ORB_create(nfeatures = 8000, scoreType=cv2.ORB_FAST_SCORE)
    keypoints_template, descriptors_template = orb.detectAndCompute(template_gray, None)
    keypoints_image, descriptors_image = orb.detectAndCompute(image_gray, None)

    # Step 4: So khớp descriptor sử dụng BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_template, descriptors_image)
    matches = sorted(matches, key=lambda x: x.distance)  # xắp xếp các điểm khớp theo khoảng cách tăng dần

    # Step 5: Trích xuất các điểm khớp
    src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_image[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Step 6: Tìm ma trận đồng nhất (homography) sử dụng RANSAC để loại bỏ nhiễu
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Step 7: Chuyển tọa độ
    new_coor = []
    for i in range(len(list_coor)):
        x = list_coor[i][0]
        y = list_coor[i][1]
        x1, y1 = Change_of_basis(x, y, M, w1, h1, w2, h2)
        new_coor.append([x1, y1, list_coor[i][2], list_coor[i][3]])
    
    return new_coor

