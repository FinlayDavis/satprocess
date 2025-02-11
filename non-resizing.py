import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_otsu
import tkinter as tk 
from tkinter import *
from PIL import Image 
from PIL import ImageTk 
import os
import csv

def load_shifts(file_path: str) -> dict:
    """
    Load existing shifts with error handling

    Args:
        file_path (str): Location of the shift.csv file.

    Returns:
        dict: The x/y shift values.
    """
    shifts = {}
    try:
        if os.path.exists(file_path):
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader: 
                    if len(row) >= 3:
                        filename, shift_y, shift_x = row[0], float(row[1]), float(row[2])
                        shifts[filename] = (shift_y, shift_x)
    except Exception as e:
        print(f"Warning: Error loading shifts - {str(e)}")
    return shifts


def load_2D_array(filename: str, wavelength: int) -> np.ndarray:
    """
    Loads a single wavelength of the specified .fits file to an array.

    Args:
        filename (str): The name of the .fits file to be loaded.
        wavelength (int): The index of the wavelength to extract.

    Returns:
        np.ndarray: A 2D array of the solar image at the specified wavelength.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the specified wavelength is out of bounds.
        IndexError: If the .fits file does not have the expected 3D structure.
    """
    # Check if the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    
    try:
        with fits.open(filename) as hdul:
            # Ensure the file has the expected structure
            if len(hdul) < 2:
                raise IndexError("The .fits file does not contain the expected data extension.")
            
            data = hdul[1].data
            
            # Check if the data is 3D
            if data.ndim != 3:
                raise IndexError("The .fits file data is not 3D (expected [wavelength, x, y]).")
            
            # Check if the wavelength index is valid
            if wavelength < 0 or wavelength >= data.shape[0]:
                raise ValueError(
                    f"Wavelength index {wavelength} is out of bounds. "
                    f"Valid range is 0 to {data.shape[0] - 1}."
                )
            
            # Extract the specified wavelength
            array = data[wavelength, :, :]
            return array
    
    except Exception as e:
        # Catch any unexpected errors and re-raise with context
        raise RuntimeError(f"An error occurred while processing {filename}: {str(e)}")


def save_shifts(file_path: str, shifts: csv):
    """
    Save alignment shifts to CSV with directory creation.

    Args:
        file_path (str): Folder location to save the .csv file in.
        shifts (dict): The calculated coordinate difference between the center of this circle, and the center of the reference.
    """
    directory = os.path.dirname(file_path)
    if directory:  # Only create if path contains directories
        os.makedirs(directory, exist_ok=True)
    try:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'shift_y', 'shift_x'])
            for filename, (shift_y, shift_x) in shifts.items():
                writer.writerow([filename, shift_y, shift_x])
    except IOError as e:
        print(f"Error saving shifts: {str(e)}")


def preprocess(image_array: np.ndarray)-> np.ndarray:
    """
    Preprocessing a 2D image array so it is easier for the Hough transform to identify features.

    Args:
        image_array (np.ndarray): The 2D image array to be processed.

    Returns:
        np.ndarray: The processed image.
    """
    # Apply Gaussian blur 
    blurred = sp.gaussian_filter(image_array, sigma=2) 

    # Apply thresholding 
    thresh = threshold_otsu(blurred) 
    binary = blurred > thresh

    # Compute edge map using Canny for better performance
    edges = canny(binary, sigma=1)
    return edges


def hough_transform(edges: np.ndarray, minrad: int = 800, maxrad: int = 1000)-> np.ndarray:
    """
    Performs the circular hough transform on the inputted 2D array, looking for circles in the specified size range.

    Args:
        edges (np.ndarray): The 2D array containing the image data
        minrad (int, optional): Minumum radius of indentified circles. Defaults to 800.
        maxrad (int, optional): Maximum radius of indentified circles. Defaults to 1000.

    Returns:
        np.ndarray: An array containing all the found circles (hopefully just the one)
    """
    # Perform Circular Hough Transform for large circles
    hough_radii = np.arange(minrad, maxrad, 1)  # Look for circle radii in the range 800-1000 (in the array, the radius of the sun is 925 for this file), with a step size of 3
    hough_res = hough_circle(edges, hough_radii)

    # Identify circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    identifiedCircles = np.array(list(zip(cx, cy, radii)))
    return identifiedCircles


def transform_array(dynArray: np.ndarray, shift_x: int, shift_y: int)-> np.ndarray:
    """
    Transforms the inputted array by an x and y value.

    Args:
        dynArray (np.ndarray): The dynamic array to be transformed.
        shift_x (int): The x transform.
        shift_y (int): The y transform.

    Returns:
        np.ndarray: The transformed array.
    """
    # Transform the image by the differece, to align images.
    alignedArray = sp.shift(dynArray, shift=[shift_y, shift_x], mode="nearest")
    return alignedArray


def spatial_calibration(folder_path: str, wavelength: int = 1, shifts_file: str = 'shifts.csv', default_file: int = 0, minrad: int = 800, maxrad: int = 1000):
    """
    Main processing pipeline with validation

    Args:
        folder_path (str): The name of the folder which holds the .fits files.
        wavelength (int): .fits files 3 axes, with the z axis being wavelength. This allows the user to choose which wavelenth. Defaults to 1.
        shifts_file (str, optional): The name of the .csv file which stores the spatial offset values for each .fits file. Defaults to 'shifts.csv'.
        default_file (int, optional): The file number which is used as the reference. Defaults to 0 (first file in the folder).
        minrad (int, optional): Minumum radius of indentified circles. Defaults to 800.
        maxrad (int, optional): Maximum radius of indentified circles. Defaults to 1000.

    Returns:
        list: A list containing the transformed images

    Raises:
        ValueError: Error raised if the file path doesn't exist (or isn't referenced properly)
        RuntimeError: Error raised if the specified file path cannot be accessed
        ValueError: Error raised if there are no files with the correct file extension at the location
        ValueError: Error raised if the user specified "default file" is a number higher than the number of files
        RuntimeError: Error raised if the reference file cannot be processed.
    """

    # Validate input directory
    if not os.path.isdir(folder_path):
        raise ValueError(f"Directory {folder_path} not found")
    
    # Create the AlignedImages subfolder if it doesn't exist
    aligned_folder = os.path.join(folder_path, "AlignedImages")
    os.makedirs(aligned_folder, exist_ok=True)
    
    # Get files with multiple validation checks
    try:
        files = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(('.fits', '.jpg'))]
    except Exception as e:
        raise RuntimeError(f"Error accessing directory: {str(e)}")  
    
    if not files:
        raise ValueError("No valid image files found")
    if default_file >= len(files):
        raise ValueError("Invalid default file index")

    shifts = load_shifts(shifts_file)
    aligned_images = []

    # Process reference image
    try:
        ref_file = files[default_file]
        ref_path = os.path.join(folder_path, ref_file)
        ref_array = load_2D_array(ref_path, wavelength)
        ref_edges = preprocess(ref_array)
        ref_circles = hough_transform(ref_edges, minrad, maxrad)
        ref_centroid = ref_circles[0][:2]
    except Exception as e:
        raise RuntimeError(f"Reference processing failed: {str(e)}")

    # Process all files
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            # Validate and load file
            if not os.path.isfile(file_path):
                print(f"Skipping missing file: {file}")
                continue

            file_array = load_2D_array(file_path, wavelength)
            file_edges = preprocess(file_array)

            # Calculate or load shifts
            if file in shifts:
                shift_y, shift_x = shifts[file]
            else:
                file_circles = hough_transform(file_edges, minrad, maxrad)
                file_centroid = file_circles[0][:2]   
                shift_x = ref_centroid[0] - file_centroid[0]
                shift_y = ref_centroid[1] - file_centroid[1]
                shifts[file] = (shift_y, shift_x)

            # Align and store
            aligned = transform_array(file_array, *shifts[file])
            aligned_images.append((file, aligned))

            # Save the aligned image as a new FITS file in the AlignedImages subfolder
            if file.lower().endswith('.fits'):
                new_file_name = f"{os.path.splitext(file)[0]}_aligned.fits"
                new_file_path = os.path.join(aligned_folder, new_file_name)
                save_aligned_fits(file_path, new_file_path, aligned, shift_x, shift_y)
            
        except Exception as e:
            print(f"Skipping {file} due to error: {str(e)}")
            continue

    # Save results
    try:
        save_shifts(shifts_file, shifts)
    except Exception as e:
        print(f"Warning: Failed to save shifts - {str(e)}")

    return aligned_images

def save_aligned_fits(original_path: str, new_path: str, aligned_data, shift_x: float, shift_y: float):
    """
    Save the aligned data as a new FITS file with updated header information.

    Args:
        original_path (str): Path to the original FITS file.
        new_path (str): Path to save the new aligned FITS file.
        aligned_data: The aligned image data.
        shift_x (float): The x-axis shift applied.
        shift_y (float): The y-axis shift applied.
    """
    # Read the original FITS file
    with fits.open(original_path) as hdul:
        # Update the header with the shift information
        hdul[0].header['SHIFT_X'] = (shift_x, 'X-axis shift applied during alignment')
        hdul[0].header['SHIFT_Y'] = (shift_y, 'Y-axis shift applied during alignment')
        
        # Replace the data with the aligned data
        hdul[0].data = aligned_data
        
        # Save the new FITS file
        hdul.writeto(new_path, overwrite=True)

    

def extract_centered_region(input_array: np.ndarray, output_size: tuple = (100, 100)) -> np.ndarray:
    """
    Extracts a centered region of the specified size from the input array.

    Args:
        input_array (np.ndarray): The input array (2D or higher).
        output_size (tuple): The desired size of the output region (height, width). Default is (100, 100).

    Returns:
        np.ndarray: A new array of the specified size, centered at the middle of the input array.

    Raises:
        ValueError: If the input array is not 2D or if the output size is larger than the input array.
    """
    # Ensure the input array is 2D
    if input_array.ndim != 2:
        raise ValueError("Input array must be 2D.")
    
    # Get the dimensions of the input array
    input_height, input_width = input_array.shape
    
    # Get the desired output dimensions
    output_height, output_width = output_size
    
    # Check if the output size is larger than the input array
    if output_height > input_height or output_width > input_width:
        raise ValueError("Output size cannot be larger than the input array dimensions.")
    
    # Calculate the center of the input array
    center_y = input_height // 2
    center_x = input_width // 2
    
    # Calculate the starting and ending indices for the centered region
    start_y = center_y - output_height // 2
    end_y = start_y + output_height
    start_x = center_x - output_width // 2
    end_x = start_x + output_width
    
    # Handle edge cases where the region goes out of bounds
    if start_y < 0:
        start_y = 0
        end_y = output_height
    if start_x < 0:
        start_x = 0
        end_x = output_width
    if end_y > input_height:
        start_y = input_height - output_height
        end_y = input_height
    if end_x > input_width:
        start_x = input_width - output_width
        end_x = input_width
    
    # Extract the centered region
    centered_region = input_array[start_y:end_y, start_x:end_x]
    
    return centered_region

def analyze_aligned_images(folder_path: str, output_size: tuple = (100, 100)):
    """
    Loads the aligned .fits files from the AlignedImages subfolder, extracts the centered region,
    orders the pixels by intensity, and retrieves the coordinates of the median 10%.

    Args:
        folder_path (str): The path to the folder containing the original and aligned images.
        output_size (tuple): The size of the centered region to extract. Default is (100, 100).
    """
    # Path to the AlignedImages subfolder
    aligned_folder = os.path.join(folder_path, "AlignedImages")
    
    # Check if the AlignedImages folder exists
    if not os.path.isdir(aligned_folder):
        raise ValueError(f"AlignedImages folder not found in {folder_path}")
    
    # Get all aligned .fits files
    aligned_files = [f for f in os.listdir(aligned_folder) if f.lower().endswith('.fits')]
    
    if not aligned_files:
        print("No aligned .fits files found in the AlignedImages folder.")
        return
    
    # Load each aligned file, extract the centered region, and analyze/visualize
    for file in aligned_files:
        file_path = os.path.join(aligned_folder, file)
        try:
            # Load the aligned image
            with fits.open(file_path) as hdul:
                aligned_data = hdul[0].data
            
            # Extract the centered region
            centered_region = extract_centered_region(aligned_data, output_size)
            
            # Perform analysis or visualization
            print(f"Processing {file}:")
            print(f"Centered region shape: {centered_region.shape}")
            
            # Flatten the centered region and get pixel intensities
            flat_intensities = centered_region.flatten()
            
            # Sort the intensities in ascending order
            sorted_indices = np.argsort(flat_intensities)
            sorted_intensities = flat_intensities[sorted_indices]
            
            # Calculate the range for the median 10%
            total_pixels = len(sorted_intensities)
            median_10_start = int(total_pixels * 0.45)  # Start of median 10%
            median_10_end = int(total_pixels * 0.55)    # End of median 10%
            
            # Get the median 10% intensities
            median_10_intensities = sorted_intensities[median_10_start:median_10_end]
            
            # Get the coordinates of the median 10% pixels
            median_10_indices = sorted_indices[median_10_start:median_10_end]
            median_10_coords = np.unravel_index(median_10_indices, centered_region.shape)
            
            # Visualize the centered region with median 10% pixels highlighted
            plt.figure(figsize=(6, 6))
            plt.imshow(centered_region, cmap='gray')
            
            # Overlay the median 10% pixels
            plt.scatter(median_10_coords[1], median_10_coords[0], c='red', s=10, label='Median 10%')
            plt.title(f"Centered Region: {file}")
            plt.legend()
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

### Test Usage
if __name__ == "__main__":
    folder_path = 'TestImages'
    shifts_file_name = 'shifts.csv'
    shifts_file = os.path.join(folder_path, shifts_file_name)
    
    try:
        # Perform spatial calibration and save aligned images
        aligned_images = spatial_calibration(folder_path, 40, shifts_file, 0, 800, 1000)
        
        if not aligned_images:
            print("No images processed successfully")
        else:
            # Analyze the aligned images
            analyze_aligned_images(folder_path, output_size=(100, 100))
    
    except Exception as e:
        print(f"Critical error: {str(e)}")



