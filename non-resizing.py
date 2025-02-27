import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_otsu
from tkinter import *
import os
import csv
import cProfile
import pstats


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
            with open(file_path, mode="r") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        filename, shift_y, shift_x = row[0], float(row[1]), float(row[2])
                        shifts[filename] = (shift_y, shift_x)
    except Exception as e:
        print(f"Warning: Error loading shifts - {str(e)}")
    return shifts


def load_array(
    filename: str,
    mode: str = "full",
    wavelength: int = None,
    x_coord: int = None,
    y_coord: int = None,
) -> np.ndarray:
    """
    Loads data from the specified .fits file based on the mode.

    Args:
        filename (str): The name of the .fits file to be loaded.
        mode (str): The mode of loading. Options: "full", "slice", "spectrum".
        wavelength (int): The index of the wavelength to extract (required for "slice" mode).
        x_coord (int): The x-coordinate of the pixel (required for "spectrum" mode).
        y_coord (int): The y-coordinate of the pixel (required for "spectrum" mode).

    Returns:
        np.ndarray: The loaded data.

    Raises:
        ValueError: If the mode or parameters are invalid.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    with fits.open(filename) as hdul:
        if len(hdul) < 2:
            raise IndexError("The .fits file does not contain the expected data extension.")

        data = hdul[1].data

        if data.ndim != 3:
            raise IndexError("The .fits file data is not 3D (expected [wavelength, x, y]).")

        if mode == "full":
            return data
        elif mode == "slice":
            if wavelength is None:
                raise ValueError("Wavelength must be specified for 'slice' mode.")
            if wavelength < 0 or wavelength >= data.shape[0]:
                raise ValueError(
                    f"Wavelength index {wavelength} is out of bounds. "
                    f"Valid range is 0 to {data.shape[0] - 1}."
                )
            return data[wavelength, :, :]
        elif mode == "spectrum":
            if x_coord is None or y_coord is None:
                raise ValueError("x_coord and y_coord must be specified for 'spectrum' mode.")
            if x_coord < 0 or x_coord >= data.shape[1] or y_coord < 0 or y_coord >= data.shape[2]:
                raise ValueError(
                    f"Coordinates ({x_coord}, {y_coord}) are out of bounds. "
                    f"Valid ranges are x: 0 to {data.shape[1] - 1}, y: 0 to {data.shape[2] - 1}."
                )
            return data[:, x_coord, y_coord]
        else:
            raise ValueError(f"Invalid mode: {mode}. Options are 'full', 'slice', 'spectrum'.")


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
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "shift_y", "shift_x"])
            for filename, (shift_y, shift_x) in shifts.items():
                writer.writerow([filename, shift_y, shift_x])
    except IOError as e:
        print(f"Error saving shifts: {str(e)}")


def preprocess(image_array: np.ndarray) -> np.ndarray:
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


def hough_transform(edges: np.ndarray, minrad: int = 800, maxrad: int = 1000) -> np.ndarray:
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
    hough_radii = np.arange(
        minrad, maxrad, 1
    )  # Look for circle radii in the range 800-1000 (in the array, the radius of the sun is 925 for this file), with a step size of 3
    hough_res = hough_circle(edges, hough_radii)

    # Identify circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    identifiedCircles = np.array(list(zip(cx, cy, radii)))
    return identifiedCircles


def transform_array(dynArray: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
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


def spatial_calibration(
    folder_path: str,
    wavelength: int = 1,
    shifts_file: str = "shifts.csv",
    default_file: int = 0,
    minrad: int = 800,
    maxrad: int = 1000,
):
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
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".fits", ".jpg"))]
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
        ref_array = load_array(ref_path, "slice", wavelength)
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

            file_array = load_array(file_path, "slice", wavelength)
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
            if file.lower().endswith(".fits"):
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


def save_aligned_fits(
    original_path: str, new_path: str, aligned_data, shift_x: float, shift_y: float
):
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
        hdul[0].header["SHIFT_X"] = (shift_x, "X-axis shift applied during alignment")
        hdul[0].header["SHIFT_Y"] = (shift_y, "Y-axis shift applied during alignment")

        # Replace the data with the aligned data
        hdul[0].data = aligned_data

        # Save the new FITS file
        hdul.writeto(new_path, overwrite=True)


import numpy as np
import matplotlib.pyplot as plt

def median_pixels(
    input_array: np.ndarray, center_x: int, center_y: int, square_size: int = 100, percentage: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts a square region from the input array and returns the coordinates of the median percentage of pixels
    relative to the full image.

    Args:
        input_array (np.ndarray): The input 2D array.
        center_x (int): The x-coordinate for the center of the square region.
        center_y (int): The y-coordinate for the center of the square region.
        square_size (int): The size of the square region to extract (default is 100x100 pixels).
        percentage (int): The percentage of pixels to extract (e.g., 1 for median 1%).

    Returns:
        tuple: Arrays of x and y coordinates for the median percentage of pixels, relative to the full image.

    Raises:
        ValueError: If the input array is not 2D or if the square region is out of bounds.
    """
    # Check to make sure the array is 2D (a flat image)
    if input_array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    # Get the dimensions of the input array
    input_height, input_width = input_array.shape

    # Calculate the starting and ending indices for the square region
    half_size = square_size // 2
    start_y = center_y - half_size
    end_y = center_y + half_size
    start_x = center_x - half_size
    end_x = center_x + half_size

    # Ensure the region is within bounds
    start_y = max(0, start_y)
    end_y = min(input_height, end_y)
    start_x = max(0, start_x)
    end_x = min(input_width, end_x)

    # Extract the square region
    square_region = input_array[start_y:end_y, start_x:end_x]

    # Flatten the pixels into a 1D array, and then sort by intensity
    flat_intensities = square_region.flatten()
    sorted_indices = np.argsort(flat_intensities)
    sorted_intensities = flat_intensities[sorted_indices]

    # Calculate the range for the median percentage
    num_of_pixels = len(sorted_intensities)
    med_start = int(num_of_pixels * (50 - percentage / 2) / 100)  # Start of median percentage
    med_end = int(num_of_pixels * (50 + percentage / 2) / 100)    # End of median percentage

    print(num_of_pixels)
    print(med_start)
    print(med_end)

    # Select the median percentage of pixels
    median_pixels_indices = sorted_indices[med_start:med_end]

    # Get the coordinates of the median pixels relative to the full image
    y_coords, x_coords = np.unravel_index(median_pixels_indices, square_region.shape)
    median_y_full = y_coords + start_y
    median_x_full = x_coords + start_x

    # Debug: Plot the image with the square region and median pixels
    plt.imshow(input_array, cmap='gray')
    plt.scatter(median_x_full, median_y_full, c='red', s=10, label='Median Pixels')
    plt.scatter(center_x, center_y, c='blue', s=50, marker='x', label='Center')
    plt.title("Square Region with Median Pixels")
    plt.legend()
    plt.show()

    return median_x_full, median_y_full

def wavelength_calibration(folder_path: str, square_size: int = 100, percentage: int = 1):
    # Check if the AlignedImages folder exists
    aligned_folder = os.path.join(folder_path, "AlignedImages")
    
    if not os.path.isdir(aligned_folder):
        raise ValueError(f"AlignedImages folder not found in {folder_path}")

    # Get all aligned .fits files
    aligned_files = [f for f in os.listdir(aligned_folder) if f.lower().endswith(".fits")]

    if not aligned_files:
        print("No aligned .fits files found in the AlignedImages folder.")
        return

    rolling_averages = []  # List to store rolling averages for each file
    decompressed_data_cache = {}  # Cache decompressed data

    for file in aligned_files:
        file_path = os.path.join(aligned_folder, file)
        try:
            # Decompress and cache the data
            if file_path not in decompressed_data_cache:
                aligned_data = load_array(file_path, "slice", 1)
                decompressed_data_cache[file_path] = aligned_data
            else:
                aligned_data = decompressed_data_cache[file_path]
            
            edges = preprocess(aligned_data)
            results = hough_transform(edges)

            # Check if any circles were detected
            if results.size == 0:
                print(f"No circles detected in {file}. Skipping.")
                continue

            # Use the first detected circle
            center_x, center_y, _ = results[0]

            # Extract the centered region and get the median percentage coordinates
            median_10_x, median_10_y = median_pixels(aligned_data, center_x, center_y, square_size=square_size, percentage=percentage)

            # Initialize rolling average for this file
            rolling_avg = None
            num_pixels = 0

            # Process each median pixel
            for i in range(len(median_10_x)):
                # Load the 1D spectrum for the specified pixel
                wavelength_data = load_array(file_path, mode="spectrum", x_coord=median_10_x[i], y_coord=median_10_y[i])
                
                # Update the rolling average
                if rolling_avg is None:
                    rolling_avg = wavelength_data  # Initialize with the first spectrum
                else:
                    rolling_avg = (rolling_avg * num_pixels + wavelength_data) / (num_pixels + 1)

                # Increment the number of pixels processed
                num_pixels += 1

            # Print the final rolling average for the file
            print(f"Final rolling average for {file_path}: {rolling_avg}")

            # Add this rolling average to the list
            rolling_averages.append((file, rolling_avg))

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Plot the rolling averages for all files
    plt.figure(figsize=(10, 6))
    for file, spectrum in rolling_averages:
        plt.plot(spectrum, label=file)  # Plot each spectrum with a label

    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Rolling Average Spectra for All Files')
    plt.legend()
    plt.grid(True)
    plt.show()

     
def wavelength_calibration_profiled():
    # The folder, and how wide the range of median pixels is (1% is 100 pixels)
    wavelength_calibration("TestImages", 100, 1)


### Test Usage
if __name__ == "__main__":
    folder_path = "TestImages"
    shifts_file_name = "shifts.csv"
    shifts_file = os.path.join(folder_path, shifts_file_name)

    try:
        # Perform spatial calibration and save aligned images
        # Directs to the folder path listed above, using a middle wavelength (40), starting at the first file and with a radius range of 200
        aligned_images = spatial_calibration(folder_path, 40, shifts_file, 0, 800, 1000)

        if not aligned_images:
            print("No images processed successfully")
        else:

            # Profile the function (Testing the timings on which process is slowest)
            profiler = cProfile.Profile()
            profiler.enable()

            # This actually runs the wavelength calibration
            wavelength_calibration_profiled()
            profiler.disable()

            # Write the profiling results to a file
            with open("profile_stats.txt", "w") as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.strip_dirs()
                stats.sort_stats(pstats.SortKey.TIME)
                stats.print_stats()

    except Exception as e:
        print(f"Critical error: {str(e)}")
