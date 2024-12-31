import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_otsu
import os
import csv

def save_shifts(file_path, shifts):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'shift_y', 'shift_x'])
        for filename, (shift_y, shift_x) in shifts.items():
            writer.writerow([filename, shift_y, shift_x])

def load_shifts(file_path):
    shifts = {}
    try:
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                filename, shift_y, shift_x = row
                shifts[filename] = (float(shift_y), float(shift_x))
    except FileNotFoundError:
        pass  # No shifts file exists yet
    return shifts


def circleIdentify(filename: str):
    # Load file from documents
    #filepath1 = "RSM20241001T013504_0002_HA.fits"

    with fits.open(filename) as hdul:
        data = hdul[1].data
        # print(data.shape) # Print the shape of the compressed image data
        # Store the 70th row as a 2D array
        array = data[40, :, :]
        # print(array.shape)  # should be 2313 by 2304 array
    
    return array
   


# Explain Gaussian blur, thresholding and Canny edge map

def preprocess(image_array: np.ndarray):

    # Apply Gaussian blur 
    blurred = sp.gaussian_filter(image_array, sigma=2) 

    # Apply thresholding 
    thresh = threshold_otsu(blurred) 
    binary = blurred > thresh

    # Compute edge map using Canny for better performance
    edges = canny(binary, sigma=1)
    return edges


# How does the Hough transform work?

def houghTransform(edges: np.ndarray, array: np.ndarray):
    # Perform Circular Hough Transform for large circles
    hough_radii = np.arange(800, 1000, 1)  # Look for circle radii in the range 800-1000 (in the array, the radius of the sun is 925 for this file), with a step size of 3
    hough_res = hough_circle(edges, hough_radii)

    # Identify circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    circles = np.array(list(zip(cx, cy, radii)))
    print(type(circles))
    return circles
# rename circles to be clearer

# Transforming the second image to be lined up with the first

def alignImages(dynArray: np.ndarray, shift_x, shift_y):
    # Transform the image by the differece, to align images.
    alignedArray = sp.shift(dynArray, shift=[shift_y, shift_x], mode="nearest")
    return alignedArray

def process_and_align_images(folder_path, shifts_file, default_file = 0):
    files = [f for f in os.listdir(folder_path) if f.endswith('.fits') or f.endswith('.jpg')]  # Adjust extensions as needed
    
    if not files:
        raise ValueError("No image files found in the folder.")
    
    # Load any cached transform values 
    shifts = load_shifts(shifts_file)

    # Read the first image as the reference
    ref_file = files[default_file]
    ref_path = os.path.join(folder_path, ref_file)
    ref_array = circleIdentify(ref_path)
    ref_edges = preprocess(ref_array)
    ref_circles = houghTransform(ref_edges, ref_array)
    ref_centroid = ref_circles[0][:2]
    
    aligned_images = []
    
    # Process and align all images
    for file in files:
        file_path = os.path.join(folder_path, file)
        file_array = circleIdentify(file_path)
        file_edges = preprocess(file_array)
        
        # Check if file has already been processed
        if file in shifts:
            shift_x, shift_y = shifts[file]
        else:
            file_circles = houghTransform(file_edges, file_array)
            file_centroid = file_circles[0][:2]
            shift_x = ref_centroid[0] - file_centroid[0]
            shift_y = ref_centroid[1] - file_centroid[1]
            shifts[file] = (shift_y, shift_x)
        
        aligned_image = alignImages(file_array, shift_y, shift_x)
        aligned_images.append((file, aligned_image))
    
    # save the shifting values for the future
    save_shifts(shifts_file, shifts)

    return aligned_images

# Example usage
folder_path = 'TestImages'
shifts_file = folder_path + '/shifts.csv'
print(shifts_file)
aligned_images = process_and_align_images(folder_path, shifts_file)

# Visualize the aligned results
fig, axes = plt.subplots(1, len(aligned_images), figsize=(15, 6))
for ax, (file, aligned_image) in zip(axes, aligned_images):
    ax.imshow(aligned_image, cmap='gray')
    ax.set_title(file)
    ax.axis('off')
plt.show()



# image1array = circleIdentify("RSM20241001T013504_0002_HA.fits")
# image1data = preprocess(image1array)

# image2array = circleIdentify("RSM20241001T013616_0003_HA.fits")
# image2data = preprocess(image2array)

# newarray = transformDynArray(image1data, image2array, image2data)
# image3data = preprocess(newarray)
# print(image3data)
