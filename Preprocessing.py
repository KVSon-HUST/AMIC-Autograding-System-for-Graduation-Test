import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import shutil


def adjust_brightness(image, factor=0.5):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor=0.5):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_saturation(image, factor=0.5):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def adjust_sharpness(image, factor=0.5):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def adjust_hue(image, shift=10):
    image = np.array(image.convert('HSV'))
    image[..., 0] = (image[..., 0].astype(int) + shift) % 180
    return Image.fromarray(image, 'HSV').convert('RGB')


def tang_do_sang_hsv(image, value):
    """
    Tăng độ sáng của ảnh bằng cách chỉnh sửa kênh V trong không gian màu HSV.
    
    Parameters:
    - image: Ảnh đầu vào (numpy array).
    - value: Giá trị tăng sáng (int, có thể âm để giảm sáng).
    
    Returns:
    - Ảnh đã tăng sáng (numpy array).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Chuyển sang HSV
    h, s, v = cv2.split(hsv)                     # Tách các kênh
    v = cv2.add(v, value)                        # Tăng giá trị độ sáng
    v = np.clip(v, 0, 255).astype(np.uint8)      # Đảm bảo giá trị nằm trong [0, 255]
    hsv = cv2.merge((h, s, v))                   # Gộp các kênh lại
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Chuyển lại BGR
    return brightened_image

def do_do_sang(image):
    """
    Đo độ sáng trung bình của một ảnh.
    
    Parameters:
    - image: Ảnh đầu vào (numpy array).
    
    Returns:
    - Độ sáng trung bình (float).
    """
    # Chuyển ảnh sang Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Tính độ sáng trung bình
    average_brightness = np.mean(gray_image)
    return average_brightness

def disnoise_image(img):
    
    denoised = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Convert to PIL image
    # denoised_image = Image.fromarray(denoised)
    return denoised

def process_images(input_folder, output_folder, input_label, output_label, threshold_value=240):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)



    # Duyệt qua toàn bộ file trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        label_file = filename.rsplit('.', 1)[0] + '.txt'
        image_path = os.path.join(input_folder, filename)
        label_path = os.path.join(input_label, label_file)

        for i in range(1, 5+1):
            # Copy image
            image_copy_name = f"{filename.rsplit('.', 1)[0]}_copy{i}.{filename.rsplit('.', 1)[1]}"
            image_copy_path = os.path.join(output_folder, image_copy_name)
            shutil.copy(image_path, image_copy_path)

            # Copy label
            label_copy_name = f"{label_file.rsplit('.', 1)[0]}_copy{i}.txt"
            label_copy_path = os.path.join(output_label, label_copy_name)
            shutil.copy(label_path, label_copy_path)

            # Kiểm tra nếu file là ảnh (dựa trên đuôi file)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Đọc ảnh
                image = cv2.imread(input_path)
                
                image2 = Image.open(input_path)
                image2 = adjust_brightness(image2, random.uniform(1.0, 1.3))
                image2 = adjust_contrast(image2, random.uniform(0.7, 1.3))
                image2 = adjust_saturation(image2, random.uniform(0.7, 1.3))

                # Tăng sáng
                brightness = do_do_sang(image)
                if brightness < 245:
                    image = tang_do_sang_hsv(image, 245 - brightness)
                # print(f"Độ sáng trung bình của ảnh là: {brightness:.2f}")
                

                # Chuyển đổi ảnh sang gray-scale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Áp dụng ngưỡng (threshold)
                _, threshold_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

                # Khử nhiễu bằng bộ lọc bilateral
                denoised_image = cv2.bilateralFilter(threshold_image, d=15, sigmaColor=100, sigmaSpace=75)  

                # Lưu ảnh đã xử lý vào thư mục đích
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, denoised_image)
                # print(f"Đã xử lý: {filename}")



# Đường dẫn thư mục đầu vào và đầu ra
input_folder = "E:/OneDrive - Hanoi University of Science and Technology/HUST/Competition/AMIC/Dataset/Train/"  # Thay bằng đường dẫn thư mục đầu vào
output_folder = "E:/OneDrive - Hanoi University of Science and Technology/HUST/Competition/AMIC/Processed_data/Train/"  # Thay bằng đường dẫn thư mục đầu ra

input_label = 'E:/OneDrive - Hanoi University of Science and Technology/HUST/Competition/AMIC/Dataset/Labels/Labels/'
output_label = 'E:/OneDrive - Hanoi University of Science and Technology/HUST/Competition/AMIC/Processed_data/labels/'

# Gọi hàm xử lý ảnh
process_images(input_folder, output_folder, input_label, output_label)
