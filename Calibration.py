import os
import csv
import cProfile
import pstats
import concurrent.futures
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_otsu
from scipy.interpolate import interp1d
from scipy.signal import correlate


# 0.024 A for pixel spectral resolution

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
                        filename, shift_y, shift_x = (
                            row[0],
                            float(row[1]),
                            float(row[2]),
                        )
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

    with fits.open(filename, menmap = True) as hdul:
        if len(hdul) < 2:
            raise IndexError(
                "The .fits file does not contain the expected data extension."
            )

        data = hdul[1].data

        if data.ndim != 3:
            raise IndexError(
                "The .fits file data is not 3D (expected [wavelength, x, y])."
            )

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
                raise ValueError(
                    "x_coord and y_coord must be specified for 'spectrum' mode."
                )
            if (
                x_coord < 0
                or x_coord >= data.shape[1]
                or y_coord < 0
                or y_coord >= data.shape[2]
            ):
                raise ValueError(
                    f"Coordinates ({x_coord}, {y_coord}) are out of bounds. "
                    f"Valid ranges are x: 0 to {data.shape[1] - 1}, y: 0 to {data.shape[2] - 1}."
                )
            return data[:, x_coord, y_coord]
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Options are 'full', 'slice', 'spectrum'."
            )


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
    hough_radii = np.arange(minrad, maxrad, 1)
    hough_res = hough_circle(edges, hough_radii)

    # Identify circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    identifiedCircles = np.array(list(zip(cx, cy, radii)))
    print(f"Detected circles: {identifiedCircles}")  # Debug print
    return identifiedCircles



# Somewhere in the code below is a memory leak? Or some sort of overload causing the following error:
# Critical error: Reference processing failed: [WinError 1450] Insufficient system resources exist to complete the requested service

"""
def process_radius_chunk(edges, radii):
    ""Process a chunk of radii using the Hough transform.""
    return hough_circle(edges, radii)

def hough_transform_parallel(edges: np.ndarray, minrad: int = 800, maxrad: int = 1000, num_workers: int = 4) -> np.ndarray:
    ""
    Performs the circular Hough transform on the inputted 2D array in parallel.

    Args:
        edges (np.ndarray): The 2D array containing the image data.
        minrad (int): Minimum radius of identified circles.
        maxrad (int): Maximum radius of identified circles.
        num_workers (int): Number of parallel workers.

    Returns:
        np.ndarray: An array containing all the found circles.
    ""
    # Divide the radius range into chunks for parallel processing
    radius_chunks = np.array_split(np.arange(minrad, maxrad), num_workers)
    
    # Use ProcessPoolExecutor to parallelize the Hough transform
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks for each radius chunk
        futures = [executor.submit(process_radius_chunk, edges, chunk) for chunk in radius_chunks]

        # Collect results as they complete
        hough_res_chunks = []
        for future in concurrent.futures.as_completed(futures):
            hough_res_chunks.append(future.result())

    # Combine the results from all chunks
    hough_res = np.concatenate(hough_res_chunks, axis=0)

    # Identify circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, np.arange(minrad, maxrad), total_num_peaks=1)

    identified_circles = np.array(list(zip(cx, cy, radii)))
    return identified_circles

"""


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
    Spatially calibrates all the .fits files in the input folder, referenced to the default file.

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
        files = [
            f for f in os.listdir(folder_path) if f.lower().endswith((".fits", ".jpg"))
        ]
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
        print(f"Reference centroid: {ref_centroid}")  # Debug print
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


def median_pixels(
    input_array: np.ndarray,
    center_x: int,
    center_y: int,
    square_size: int = 100,
    percentage: int = 1,
) -> tuple[np.ndarray, np.ndarray, int, int]:
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
        tuple: Arrays of x and y coordinates for the median percentage of pixels, relative to the full image,
               and the starting indices (start_x, start_y) of the square region.
    """
    # Check to make sure the array is 2D (a flat image)
    if input_array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    # Get the dimensions of the input array
    input_height, input_width = input_array.shape

    # Calculate the starting and ending indices for the square region
    half_size = square_size // 2
    start_y = max(0, center_y - half_size)
    end_y = min(input_height, center_y + half_size)
    start_x = max(0, center_x - half_size)
    end_x = min(input_width, center_x + half_size)

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

    # Select the median percentage of pixels
    median_pixels_indices = sorted_indices[med_start:med_end]

    # Get the coordinates of the median pixels relative to the full image
    y_coords, x_coords = np.unravel_index(median_pixels_indices, square_region.shape)
    median_y_full = y_coords + start_y
    median_x_full = x_coords + start_x

    # Plot the image with the square region and median pixels
    # plt.imshow(input_array, cmap="gray")
    # plt.scatter(median_x_full, median_y_full, c="red", s=2.5, label="Median Pixels")
    # plt.title("Square Region with Median Pixels")
    # plt.legend()
    # plt.show()

    return median_x_full, median_y_full, start_x, start_y


def process_pixel(data_3d: np.ndarray, x: int, y: int):
    """
    Extracts the spectrum for a single pixel from a preloaded 3D array.

    Args:
        data_3d (np.ndarray): The 3D array containing the wavelength data (shape: [wavelength, x, y]).
        x (int): x-coordinate of the pixel.
        y (int): y-coordinate of the pixel.

    Returns:
        np.ndarray: 1D array of the wavelength for that pixel.
    """
    try:
        # Extract the spectrum for the specified pixel
        wavelength_data = data_3d[:, x, y]
        return wavelength_data
    except Exception as e:
        print(f"Error processing pixel ({x}, {y}): {str(e)}")
        return None
    

def process_file(file_path, square_size, percentage, max_workers):
    """
    Processes a single file and calculates the rolling average spectrum for the median pixels.

    Args:
        file_path (str): Path to the .fits file.
        square_size (int): Size of the square region to extract.
        percentage (int): Percentage of median pixels to use.
        max_workers (int): Maximum number of parallel workers.

    Returns:
        tuple: (file_path, wavelengths, intensities)
    """
    try:
        # Load the aligned data and preprocess it
        aligned_data = load_array(file_path, "slice", 1)
        edges = preprocess(aligned_data)
        results = hough_transform(edges)

        # Check if any circles were detected
        if results.size == 0:
            print(f"No circles detected in {file_path}. Skipping.")
            return None

        # Use the first detected circle
        center_x, center_y, _ = results[0]

        # Load the 100x100x46 region centered at (center_x, center_y)
        region_3d = load_3d_region(file_path, center_x, center_y, square_size)
        if region_3d is None:
            return None

        # Extract the median pixels and the starting indices of the square region
        median_x_full, median_y_full, start_x, start_y = median_pixels(aligned_data, center_x, center_y, square_size, percentage)

        # Convert absolute coordinates to relative coordinates within the region_3d array
        median_x_relative = median_x_full - start_x
        median_y_relative = median_y_full - start_y

        # Initialize rolling average for this file
        rolling_avg = None
        num_pixels = 0

        # Use ProcessPoolExecutor with limited workers to reduce CPU usage
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each pixel
            futures = [
                executor.submit(process_pixel, region_3d, x, y)
                for x, y in zip(median_x_relative, median_y_relative)
            ]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                wavelength_data = future.result()
                if wavelength_data is not None:
                    # Update the rolling average
                    if rolling_avg is None:
                        rolling_avg = wavelength_data  # Initialize with the first spectrum
                    else:
                        rolling_avg = (rolling_avg * num_pixels + wavelength_data) / (num_pixels + 1)

                    # Increment the number of pixels processed
                    num_pixels += 1

        print(f"Final rolling average for {file_path}: {rolling_avg}")

        # Generate wavelengths (indices or actual wavelengths if available)
        wavelengths = np.arange(len(rolling_avg))  # Use indices as wavelengths
        return file_path, wavelengths, rolling_avg

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None
    

def load_3d_region(file_path: str, center_x: int, center_y: int, square_size: int = 100):
    """
    Loads a 3D region (wavelength, x, y) from a .fits file centered at (center_x, center_y).

    Args:
        file_path (str): Path to the .fits file.
        center_x (int): x-coordinate of the center of the region.
        center_y (int): y-coordinate of the center of the region.
        square_size (int): Size of the square region to extract (default is 100x100).

    Returns:
        np.ndarray: A 3D array of shape [wavelength, square_size, square_size].
    """
    try:
        with fits.open(file_path, memmap=True) as hdul:
            data = hdul[1].data  # Load the 3D data

            # Calculate the bounds of the square region
            half_size = square_size // 2
            start_x = max(0, center_x - half_size)
            end_x = min(data.shape[2], center_x + half_size)
            start_y = max(0, center_y - half_size)
            end_y = min(data.shape[1], center_y + half_size)

            # Extract the 3D region
            region_3d = data[:, start_y:end_y, start_x:end_x]
            return region_3d
    except Exception as e:
        print(f"Error loading 3D region from {file_path}: {str(e)}")
        return None


def wavelength_calibration(folder_path: str, square_size: int = 100, percentage: int = 1):
    """
    Performs wavelength calibration by aligning spectra from all files to the first file's spectrum.

    Args:
        folder_path (str): Path to the folder containing aligned .fits files.
        square_size (int): Size of the square region to extract.
        percentage (int): Percentage of median pixels to use.
    """
    # Check if the AlignedImages folder exists
    aligned_folder = os.path.join(folder_path, "AlignedImages")

    if not os.path.isdir(aligned_folder):
        raise ValueError(f"AlignedImages folder not found in {folder_path}")

    # Get all aligned .fits files
    aligned_files = [f for f in os.listdir(aligned_folder) if f.lower().endswith(".fits")]

    if not aligned_files:
        print("No aligned .fits files found in the AlignedImages folder.")
        return

    # Process the first file to use as the reference spectrum
    ref_file = aligned_files[0]
    ref_path = os.path.join(aligned_folder, ref_file)
    print(f"Using {ref_file} as the reference spectrum.")

    ref_result = process_file(ref_path, square_size, percentage, max_workers=4)
    if ref_result is None:
        print(f"Failed to process reference file {ref_file}. Exiting.")
        return

    ref_file_name, ref_wavelengths, ref_intensities = ref_result

    # Store the reference spectrum
    original_spectra = [(ref_file_name, ref_wavelengths, ref_intensities)]
    aligned_spectra = [(ref_file_name, ref_wavelengths, ref_intensities)]

    # Process the remaining files
    for file in aligned_files[1:]:  # Skip the first file (already processed as reference)
        file_path = os.path.join(aligned_folder, file)
        result = process_file(file_path, square_size, percentage, max_workers=4)
        if result is not None:
            file_name, target_wavelengths, target_intensities = result

            # Save original spectrum
            original_spectra.append((file_name, target_wavelengths, target_intensities))

            # Align the target spectrum to the reference spectrum
            aligned_intensities = align_spectrum(ref_wavelengths, ref_intensities, target_wavelengths, target_intensities)
            aligned_spectra.append((file_name, ref_wavelengths, aligned_intensities))

            # Save the aligned spectrum to a new .fits file
            save_aligned_spectrum(file_path, aligned_intensities, ref_wavelengths)

    # Plot original spectra
    plt.figure(figsize=(10, 6))
    for file, wavelengths, intensities in original_spectra:
        plt.plot(wavelengths, intensities, label=file)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Original Spectra")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, "original_spectra.png"))
    plt.close()

    # Plot aligned spectra
    plt.figure(figsize=(10, 6))
    for file, wavelengths, intensities in aligned_spectra:
        plt.plot(wavelengths, intensities, label=file)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Aligned Spectra")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, "aligned_spectra.png"))
    plt.close()

    print(f"Plots saved in {folder_path}.")


def align_spectrum(ref_wavelengths, ref_intensities, target_wavelengths, target_intensities):
    """
    Aligns the target spectrum to the reference spectrum using cross-correlation and interpolation.
    """
    # Plot the original spectra
    plt.figure(figsize=(10, 6))
    plt.plot(ref_wavelengths, ref_intensities, label="Reference Spectrum")
    plt.plot(target_wavelengths, target_intensities, label="Target Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Original Spectra")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate the shift required to align the target spectrum to the reference spectrum
    shift = calculate_spectral_shift(ref_intensities, target_intensities)

    
    if shift == 0:
        print("Spectra are already aligned. No shift applied.")
        return target_intensities  # Return the original target intensities
    
    # Shift the target spectrum
    shifted_intensities = np.roll(target_intensities, shift)

    # Create an interpolation function for the shifted target spectrum
    interpolate_func = interp1d(target_wavelengths, shifted_intensities, kind='linear', fill_value="extrapolate")

    # Interpolate shifted intensities onto the reference wavelength grid
    aligned_intensities = interpolate_func(ref_wavelengths)

    # Plot the aligned spectra
    plt.figure(figsize=(10, 6))
    plt.plot(ref_wavelengths, ref_intensities, label="Reference Spectrum")
    plt.plot(ref_wavelengths, aligned_intensities, label="Aligned Target Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Aligned Spectra")
    plt.legend()
    plt.grid(True)
    plt.show()

    return aligned_intensities


def normalize_spectrum(spectrum):
    """
    Normalize the spectrum to have zero mean and unit variance.
    """
    return (spectrum - np.mean(spectrum)) / np.std(spectrum)

def calculate_spectral_shift(ref_intensities, target_intensities):
    """
    Calculates the shift required to align the target spectrum to the reference spectrum using cross-correlation.
    """
    # Normalize the spectra
    ref_normalized = normalize_spectrum(ref_intensities)
    target_normalized = normalize_spectrum(target_intensities)

    # Use a larger subset of the spectra for cross-correlation
    subset_size = min(len(ref_normalized), len(target_normalized))  # Use the full spectrum
    ref_subset = ref_normalized[:subset_size]
    target_subset = target_normalized[:subset_size]

    # Calculate cross-correlation
    correlation = correlate(ref_subset, target_subset, mode='full')
    shift = np.argmax(correlation) - (len(ref_subset) - 1)

    print(f"Cross-correlation result: {correlation}")  # Debug print
    print(f"Calculated shift: {shift}")  # Debug print
    return shift

def save_aligned_spectrum(file_path: str, aligned_intensities: np.ndarray, ref_wavelengths: np.ndarray):
    """
    Saves the aligned spectrum to a new .fits file.

    Args:
        file_path (str): Path to the original .fits file.
        aligned_intensities (np.ndarray): Aligned intensities of the spectrum.
        ref_wavelengths (np.ndarray): Wavelengths of the reference spectrum.
    """
    os.path.join(file_path, "interpolated")
    with fits.open(file_path) as hdul:
        # Update the data with the aligned intensities
        hdul[1].data = aligned_intensities

        # Update the header with the new wavelength information
        hdul[1].header["WAVE"] = (str(ref_wavelengths.tolist()), "Reference wavelengths")

        # Save the new .fits file
        new_file_path = file_path.replace(".fits", "_interpolated.fits")
        hdul.writeto(new_file_path, overwrite=True)

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
        aligned_images = spatial_calibration(folder_path, 40, shifts_file, 0, 900, 950)

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
