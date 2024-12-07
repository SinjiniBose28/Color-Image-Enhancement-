import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PSNR Calculation
def calculate_psnr(original, enhanced):
    mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')  # Infinite PSNR when images are identical
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr

# Entropy Calculation
def calculate_entropy(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = np.sum(histogram)
    probabilities = histogram / total_pixels
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# SSIM Calculation
def calculate_ssim(original, enhanced):
    mu_x = np.mean(original)
    mu_y = np.mean(enhanced)
    sigma_x = np.var(original)
    sigma_y = np.var(enhanced)
    sigma_xy = np.mean((original - mu_x) * (enhanced - mu_y))
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    luminance = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    contrast = (2 * np.sqrt(sigma_x) * np.sqrt(sigma_y) + C2) / (sigma_x + sigma_y + C2)
    structure = (sigma_xy + C2 / 2) / (np.sqrt(sigma_x) * np.sqrt(sigma_y) + C2 / 2)

    return luminance * contrast * structure

# Fitness Calculation
def calculate_fitness(image):
    entropy = calculate_entropy(image)
    edge_strength = calculate_edge_strength(image)
    variance = np.var(image)
    fitness = 0.4 * entropy + 0.3 * edge_strength + 0.3 * variance
    return fitness

# Edge Strength Calculation
def calculate_edge_strength(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(edge_strength)

# Detail Variance Calculation
def calculate_detail_variance(image):
    edges = cv2.Canny(image, 100, 200)  # Edge map
    detail_pixels = image[edges > 0]  # Pixels corresponding to edges
    return np.var(detail_pixels) if len(detail_pixels) > 0 else 0

# Background Variance Calculation
def calculate_background_variance(image):
    edges = cv2.Canny(image, 100, 200)  # Edge map
    background_pixels = image[edges == 0]  # Pixels not corresponding to edges
    return np.var(background_pixels) if len(background_pixels) > 0 else 0

# Visualization: Image Comparisons with Difference Maps
def visualize_images_with_differences(original, enhancements, titles):
    num_images = len(enhancements)
    plt.figure(figsize=(20, 15))

    for idx, (enhanced, title) in enumerate(zip(enhancements, titles)):
        # Enhanced Image
        plt.subplot(3, num_images, idx + 1)
        plt.imshow(enhanced, cmap='gray')
        plt.title(title)
        plt.axis('off')

       
    # Original Image
    plt.subplot(3, num_images, num_images * 2 + 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Histogram Plotting
def plot_histograms(images, titles):
    plt.figure(figsize=(15, 10))
    num_images = len(images)
    
    for idx, (image, title) in enumerate(zip(images, titles)):
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.subplot(2, (num_images + 1) // 2, idx + 1)
        plt.plot(histogram, color='blue')
        plt.title(f"Histogram: {title}")
        plt.xlim([0, 256])
        plt.grid()
    
    plt.tight_layout()
    plt.show()

# Load Image
image_path = 'mount.jpg'  # Replace with your image path
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    raise FileNotFoundError("Image file not found. Please check the path.")

# Enhancement Techniques
he_image = cv2.equalizeHist(original_image)
gamma_image = cv2.convertScaleAbs(np.power(original_image / 255.0, 1.5) * 255)
contrast_stretch_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)

# Techniques and Images
techniques = ["Original", "Histogram Equalization", "Gamma Correction", "Contrast Stretching"]
images = [original_image, he_image, gamma_image, contrast_stretch_image]

# Calculate Metrics
metrics = []
for image, technique in zip(images, techniques):
    psnr = calculate_psnr(original_image, image) if technique != "Original" else None
    entropy = calculate_entropy(image)
    ssim = calculate_ssim(original_image, image) if technique != "Original" else None
    fitness = calculate_fitness(image)
    detail_variance = calculate_detail_variance(image)
    background_variance = calculate_background_variance(image)
    metrics.append([technique, psnr, entropy, ssim, fitness, detail_variance, background_variance])

# Create Results DataFrame
results_df = pd.DataFrame(metrics, columns=["Technique", "PSNR", "Entropy", "SSIM", "Fitness", "Detail Variance", "Background Variance"])
print(results_df)

# Save Results
results_df.to_csv('enhancement_results_with_variances.csv', index=False)

# Visualizations
titles = ["Histogram Equalization", "Gamma Correction", "Contrast Stretching"]
enhancements = [he_image, gamma_image, contrast_stretch_image]

# Visualize Images and Differences
visualize_images_with_differences(original_image, enhancements, titles)

# Plot Histograms
plot_histograms(images, techniques)
