import math
from PIL import Image
from PIL import Image  # Open & save
import matplotlib.pyplot as PLT  # Show image
import numpy as np


def Read_Image_to_Array(image_file_name):
    return np.array(image_file_name)


def increase_brightness(image, image_name):
    image_array = Read_Image_to_Array(image)
    lighten_image = image_array + float(100)
    lighten_image = np.clip(lighten_image, 0, 255)
    export_name = f"{image_name.split('.')[0]}_increase_brightness.{image_name.split('.')[1]}"
    Save_Image(lighten_image.astype(np.uint8), export_name)


def increase_contrast(image, image_name, scale=0.5):
    image_array = np.array(image)
    image_float = image_array.astype(float)
    contrast_adjusted_img = image_float * (1 + scale)
    np.clip(contrast_adjusted_img, 0, 255, out=contrast_adjusted_img)
    export_name = f"{image_name.split('.')[0]}_increase_contrast.{image_name.split('.')[1]}"
    Save_Image(contrast_adjusted_img.astype(np.uint8), export_name)


def flip_image(image, image_name):
    flip_vertical(image, image_name)
    flip_horizontal(image, image_name)


def flip_vertical(image, image_name):
    image_array = Read_Image_to_Array(image)
    flipped_image_array = np.flipud(image_array)
    export_name = f"{image_name.split('.')[0]}_flip_vertical.{image_name.split('.')[1]}"
    Save_Image(flipped_image_array.astype(np.uint8), export_name)


def flip_horizontal(image, image_name):
    image_array = Read_Image_to_Array(image)
    flipped_image_array = np.fliplr(image_array)
    export_name = f"{image_name.split('.')[0]}_flip_horizontal.{image_name.split('.')[1]}"
    Save_Image(flipped_image_array.astype(np.uint8), export_name)


def convert_to_grayscale(image, image_name):
    arrayImage = Read_Image_to_Array(image)
    newImage = np.dot(arrayImage, np.array([0.3, 0.59, 0.11]))
    export_name = f"{image_name.split('.')[0]}_convert_to_grayscale.{image_name.split('.')[1]}"
    Save_Image(newImage.astype(np.uint8), export_name)


def convert_to_sepia(image, image_name):
    image_array = np.array(image)
    red_channel = np.dot(image_array, np.array([0.393, 0.769,  0.189]))
    green_channel = np.dot(image_array, np.array([0.349, 0.686, 0.168]))
    blue_channel = np.dot(image_array, np.array([0.272, 0.534, 0.131]))
    sepia_array = np.stack((red_channel, green_channel, blue_channel), axis=-1)
    np.clip(sepia_array, 0, 255, out=sepia_array)
    export_name = f"{image_name.split('.')[0]}_convert_to_sepia.{image_name.split('.')[1]}"
    Save_Image(sepia_array.astype(np.uint8), export_name)


def apply_blur(image, image_name):
    array_image = np.array(image)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16
    blurred_image = convolve_2d(array_image, kernel)
    blurred_image = np.clip(blurred_image, 0, 255)
    export_name = f"{image_name.split('.')[0]}_apply_blur.{image_name.split('.')[1]}"
    Save_Image(blurred_image.astype(np.uint8), export_name)


def apply_sharpening(image, image_name):
    array_image = np.array(image)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = convolve_2d(array_image, kernel)
    sharpened_image = np.clip(sharpened_image, 0, 255)
    export_name = f"{image_name.split('.')[0]}_apply_sharpening.{image_name.split('.')[1]}"
    Save_Image(sharpened_image.astype(np.uint8), export_name)


def convolve_2d(image, kernel):
    if image.ndim == 2:  # Grayscale image
        return convolve(image, kernel)
    elif image.ndim == 3:  # Color image
        return np.dstack([convolve(image[:, :, i], kernel) for i in range(image.shape[2])])
    else:
        raise ValueError("Unsupported number of dimensions for the image.")


def convolve(signal, kernel):
    return np.array([[np.sum(signal[i:i+3, j:j+3] * kernel) for j in range(signal.shape[1] - 2)] for i in range(signal.shape[0] - 2)])


def crop_from_center(image, image_name):
    image_array = np.array(image)
    height, width = image_array.shape[:2]
    target_size = min(height, width) // 2
    top = (height - target_size) // 2
    bottom = top + target_size
    left = (width - target_size) // 2
    right = left + target_size
    cropped_image_array = image_array[top:bottom, left:right, :]
    export_name = f"{image_name.split('.')[0]}_crop_from_center.{image_name.split('.')[1]}"
    Save_Image(cropped_image_array, export_name)


def crop_into_circle(image, image_name):
    image_array = np.array(image)
    center_x, center_y = image_array.shape[1] // 2, image_array.shape[0] // 2
    radius = min(center_x, center_y)
    y, x = np.ogrid[:image_array.shape[0], :image_array.shape[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    cropped_image_array = np.zeros_like(image_array)
    cropped_image_array[mask] = image_array[mask]
    export_name = f"{image_name.split('.')[0]}_crop_into_circle.{image_name.split('.')[1]}"
    Save_Image(cropped_image_array, export_name)


def Save_Image(image, export_name):
    Image.fromarray(image).save(export_name)


def take_input():
    image_name = input("Nhập tên ảnh : ")
    option = -1
    print("1 Thay đổi độ sáng cho ảnh ")
    print("2 Thay đổi độ tương phản ")
    print("3 Lật ảnh (ngang - dọc) ")
    print("4 Chuyển đổi ảnh RGB thành ảnh xám ")
    print("5 Chuyển đổi ảnh RGB thành ảnh sepia ")
    print("6 Làm mờ ")
    print("7 Làm sắc nét ảnh ")
    print("8 Cắt ảnh theo kích thước (cắt ở trung tâm) ")
    print("9 Cắt ảnh theo khung hình tròn  ")

    while (option < 1 or option > 9):
        option = int(input("Lựa chọn chức năng (1-9): "))
    return option, image_name


def handle_function(option, image, image_name):
    switch_case = {
        1: increase_brightness,
        2: increase_contrast,
        3: flip_image,
        4: convert_to_grayscale,
        5: convert_to_sepia,
        6: apply_blur,
        7: apply_sharpening,
        8: crop_from_center,
        9: crop_into_circle,
    }
    select_function = switch_case.get(option)
    select_function(image, image_name)


def main():
    option, image_name = take_input()
    image = Image.open(image_name)
    handle_function(option, image, image_name)


main()
