import os
import numpy as np
from PIL import Image


def kmeans(img_1d, k_clusters, max_iter, init_centroids='random'):
    # Initialize centroids and labels
    centroids = initialize_centroids(img_1d, k_clusters, init_centroids)
    labels = np.full(img_1d.shape[0], -1)
    threshold = 1e-7

    # Run K-means iterations
    for _ in range(max_iter):
        labels = label_pixels(img_1d, centroids)
        old_centroids = centroids.copy()
        centroids = update_centroids(img_1d, labels, k_clusters)

        distance_squared = np.sum((centroids - old_centroids) ** 2)
        if distance_squared < threshold:
            break

    return centroids, labels


def initialize_centroids(img, k_clusters, init_type):
    if init_type == 'random':
        # Randomly select unique pixels as initial centroids
        unique_pixels = np.unique(img, axis=0)
        if len(unique_pixels) < k_clusters:
            raise ValueError(
                "Number of unique pixels in the image is less than k_clusters.")
        np.random.seed(0)
        indices = np.random.choice(
            len(unique_pixels), k_clusters, replace=False)
        return unique_pixels[indices]
    elif init_type == 'in_pixels':
        # Randomly select pixels from the image as initial centroids
        if len(img) < k_clusters:
            raise ValueError(
                "Number of pixels in the image is less than k_clusters.")
        unique_indices = np.unique(np.arange(len(img)))
        if len(unique_indices) < k_clusters:
            raise ValueError(
                "Number of unique indices in the image is less than k_clusters.")
        np.random.seed(0)
        indices = np.random.choice(unique_indices, k_clusters, replace=False)
        return img[indices]


def label_pixels(img, centroids):
    num_pixels = img.shape[0]
    num_centroids = centroids.shape[0]

    # Calculate distances between pixels and centroids
    distances = np.linalg.norm(img[:, np.newaxis, :] - centroids, axis=2)

    # Assign labels based on the closest centroid
    labels = np.argmin(distances, axis=1)

    return labels


def update_centroids(img, labels, k_clusters):
    num_channels = img.shape[1]
    centroids = np.zeros((k_clusters, num_channels))

    # Count the number of pixels in each cluster
    cluster_counts = np.bincount(labels, minlength=k_clusters)

    # Sum the pixel values for each cluster
    np.add.at(centroids, labels, img)

    # Divide each centroid by the number of pixels in the cluster
    mask = cluster_counts != 0
    centroids[mask] /= cluster_counts[mask][:, np.newaxis]

    return centroids


def get_user_inputs():
    while True:
        image_name = input("Enter the image name: ")
        if not os.path.isfile(image_name):
            print("Error: File not found.")
            continue
        image = Image.open(image_name)
        image = np.array(image)
        flat_image = image.reshape(
            image.shape[0] * image.shape[1], image.shape[2])
        break

    while True:
        k_clusters = input("Enter the number of clusters (k_clusters > 0): ")
        try:
            k_clusters = int(k_clusters)
            if k_clusters <= 0:
                raise ValueError
            break
        except ValueError:
            print("Error: Invalid number of clusters. Please enter a positive integer.")

    while True:
        max_iter = input(
            "Enter the maximum number of iterations (max_iter > 0): ")
        try:
            max_iter = int(max_iter)
            if max_iter <= 0:
                raise ValueError
            break
        except ValueError:
            print(
                "Error: Invalid maximum number of iterations. Please enter a positive integer.")

    while True:
        init_centroids = input(
            "Enter the initialization method for centroids (random/in_pixels): ")
        if init_centroids.lower() not in ['random', 'in_pixels']:
            print(
                "Error: Invalid initialization method. Please choose 'random' or 'in_pixels'.")
            continue
        break

    while True:
        save_format = input("Enter the save format (png/pdf): ")
        if save_format.lower() not in ['png', 'pdf']:
            print("Error: Invalid save format.")
            continue
        save_format = save_format.lower()
        break

    return flat_image, k_clusters, max_iter, init_centroids, image_name.split(".")[0], save_format, image


def output(centroids, labels, name, extension, image, k_clusters):
    # Replace each pixel with its centroid
    result = centroids[labels]
    # Reshape to the original shape and cast to np.uint8
    result = result.reshape(image.shape).astype(np.uint8)
    # Save the image with the specified filename
    output_file = f"{name}_k{k_clusters}.{extension}"
    Image.fromarray(result).save(output_file)


def main():
    img_1d, k_clusters, max_iter, init_centroids, name, extension, image = get_user_inputs()
    centroids, labels = kmeans(img_1d, k_clusters, max_iter, init_centroids)
    output(centroids, labels, name, extension, image, k_clusters)


main()
