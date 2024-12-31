import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    # filepath1 = "RSM20241001T013504_0002_HA.fits"

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
    return circles
# rename circles to be clearer

# Transforming the second image to be lined up with the first

def alignImages(dynArray: np.ndarray, shift_x, shift_y):
    # Transform the image by the differece, to align images.
    alignedArray = sp.shift(dynArray, shift=[shift_y, shift_x], mode="nearest")
    return alignedArray

def process_and_align_images(folder_path: str, shifts_file: str, save_folder: str, original_folder: str, default_file: int = 0):
    if not os.path.exists(save_folder): 
        os.makedirs(save_folder) 
    if not os.path.exists(original_folder): 
        os.makedirs(original_folder) 

    files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]  # Adjust extensions as needed
    
    if not files:
        raise ValueError("No image files found in the folder.")
    
    # Load any cached transform values 
    shifts = load_shifts(shifts_file)

    # Read the first image as the reference
    ref_file = files[default_file]
    ref_path = os.path.join(folder_path, ref_file)
    ref_array = circleIdentify(ref_path)

    if ref_file in shifts:
        ref_shift_x, ref_shift_y = shifts[ref_file]
        ref_centroid = (ref_shift_x, ref_shift_y)
    else:   
        ref_edges = preprocess(ref_array)
        ref_circles = houghTransform(ref_edges, ref_array)
        ref_centroid = ref_circles[0][:2]
        shifts[ref_file] = (ref_centroid[1], ref_centroid[0])
    
    aligned_images = []

    save_path = os.path.join(save_folder, ref_file)
    save_aligned_images(ref_file, ref_array, save_path)
    misaligned_save_path = os.path.join(original_folder, f"{os.path.splitext(ref_file)[0]}_misaligned.png")
    plt.imsave(misaligned_save_path, ref_array, cmap='gray')

    aligned_images.append((ref_file, ref_array))
    
    # Process and align all images
    for file in files[1:]: # Skip the first file as it's the reference file
        file_path = os.path.join(folder_path, file)
        file_array = circleIdentify(file_path)
        file_edges = preprocess(file_array)
        
        misaligned_save_path = os.path.join(original_folder, f"{os.path.splitext(file)[0]}_misaligned.png")
        plt.imsave(misaligned_save_path, file_array, cmap='gray')

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
        save_path = os.path.join(save_folder, file)
        save_aligned_images(file, aligned_image, save_path)
        aligned_images.append((file, file_array))
    
    # Include ref file in aligned resulrs
    aligned_images.insert(0, (ref_file, plt.imread(ref_path)))

    # Save the shifting values for the future
    save_shifts(shifts_file, shifts)

    return aligned_images

def save_aligned_images(file, array, save_path):
    # Save new arrays as .fits data
    fits.writeto(save_path, array, overwrite=True)

    # Save a .png of the figure
    fig, ax = plt.subplots()
    ax.imshow(array, cmap='gray')
    ax.set_title(file)
    ax.axis('off')
    fig_save_path = os.path.join(save_folder, f"{os.path.splitext(file)[0]}.png")
    fig.savefig(fig_save_path)
    plt.close(fig)

def load_aligned_images(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.fits')]
    images = []
    for file in files:
        file_path = os.path.join(folder, file)
        image = fits.getdata(file_path)
        images.append((file, image))
    return images

def create_slideshow(images, interval=1000):
    fig, ax = plt.subplots()
    ax.axis('off')
    
    def update(frame):
        ax.clear()
        ax.imshow(images[frame][1], cmap='gray')
        ax.set_title(images[frame][0])
        ax.axis('off')
    
    ani = FuncAnimation(fig, update, frames=len(images), interval=interval, repeat=True)
    
    def on_key_press(event):
        if event.key=='q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()

# Example usage
folder_path = 'TestImages2'
shifts_file = folder_path + '/shifts.csv'
save_folder = folder_path + '/alignedImages'
original_folder = folder_path + '/unprocessedImages'
aligned_images = process_and_align_images(folder_path, shifts_file, save_folder, original_folder)

# Example usage
aligned_images = load_aligned_images(save_folder)
create_slideshow(aligned_images)
