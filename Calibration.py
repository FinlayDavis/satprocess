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


def hough_transform(
    edges: np.ndarray, minrad: int = 800, maxrad: int = 1000
) -> np.ndarray:
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
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=1
    )

    identifiedCircles = np.array(list(zip(cx, cy, radii)))
    print(f"Detected circles: {identifiedCircles}")  # Debug print
    return identifiedCircles


def transform_array(
    dynArray: np.ndarray, shift_x: int, shift_y: int, mode: str = "nearest"
) -> np.ndarray:
    """
    Transforms the inputted array by an x and y value.
    \nOptions for input mode according to SciPy shift docs:
    \n‘nearest’
        The input is extended by replicating the last pixel.
    \n‘constant’
        The input is extended by filling all values beyond the edge with the same constant value,
        defined by the cval parameter. No interpolation is performed beyond the edges of the input.
    \n‘grid-wrap’
        The input is extended by wrapping around to the opposite edge. Useful for preserving data.

    Args:
        dynArray (np.ndarray): The dynamic array to be transformed.
        shift_x (int): The x transform.
        shift_y (int): The y transform.
        mode (str): The mode used for the spline interpolation transformation. Defaults to "nearest".

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
    Saves aligned files in the "Spatial" folder.
    """
    # Debug print
    print("Starting spatial calibration...")

    # Ensure the folder structure exists
    ensure_folder_structure(folder_path)

    # Validate input directory
    if not os.path.isdir(folder_path):
        raise ValueError(f"Directory {folder_path} not found")

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
        print(f"Using {ref_file} as the reference spectrum.")

        ref_array = load_array(ref_path, "slice", wavelength)
        ref_edges = preprocess(ref_array)
        ref_circles = hough_transform(ref_edges, minrad, maxrad)
        ref_centroid = ref_circles[0][:2]
        print(f"Reference centroid: {ref_centroid}")  # Debug print
    except Exception as e:
        raise RuntimeError(f"Reference processing failed: {str(e)}")


    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if not os.path.isfile(file_path):
                continue

            file_array = load_array(file_path, "slice", wavelength)
            file_edges = preprocess(file_array)

            if file in shifts:
                shift_y, shift_x, wavelength_shift, cx, cy = shifts[file]
            else:
                file_circles = hough_transform(file_edges, minrad, maxrad)
                cx, cy, radius = file_circles[0]  # Get center coordinates
                shift_x = ref_centroid[0] - cx
                shift_y = ref_centroid[1] - cy
                wavelength_shift = 0
                shifts[file] = (shift_y, shift_x, wavelength_shift, cx, cy)  # Store center

            # Align and store
            aligned = transform_array(file_array, *shifts[file][:2])  # Use only shift_y and shift_x
            aligned_images.append((file, aligned))

            # Save the aligned image as a new FITS file in the "Spatial" folder
            
            if file.lower().endswith(".fits"):
                save_aligned_fits(file_path, "Spatial", aligned, shift_x, shift_y)

        except Exception as e:
            print(f"Skipping {file} due to error: {str(e)}")
            continue

    # Save results
    try:
        save_shifts(shifts_file, shifts)
    except Exception as e:
        print(f"Warning: Failed to save shifts - {str(e)}")

    return aligned_images

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
    med_start = int(
        num_of_pixels * (50 - percentage / 2) / 100
    )  # Start of median percentage
    med_end = int(
        num_of_pixels * (50 + percentage / 2) / 100
    )  # End of median percentage

    # Select the median percentage of pixels
    median_pixels_indices = sorted_indices[med_start:med_end]

    # Get the coordinates of the median pixels relative to the full image
    y_coords, x_coords = np.unravel_index(median_pixels_indices, square_region.shape)
    median_y_full = y_coords + start_y
    median_x_full = x_coords + start_x

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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check shifts.csv for existing center coordinates
        shifts_file = os.path.join(os.path.dirname(os.path.dirname(file_path)), "shifts.csv")
        shifts = load_shifts(shifts_file)
        filename = os.path.basename(file_path)
        
        if filename in shifts:
            _, _, _, center_x, center_y = shifts[filename]
            print(f"Using pre-calculated center from shifts.csv: ({center_x}, {center_y})")
        else:
            # Fall back to Hough transform if no saved center
            aligned_data = load_array(file_path, "slice", 1)
            edges = preprocess(aligned_data)
            results = hough_transform(edges)
            if results.size == 0:
                print(f"No circles detected in {file_path}")
                return None
            center_x, center_y, _ = results[0]

        # Load the 100x100x46 region centered at (center_x, center_y)
        region_3d = load_3d_region(file_path, center_x, center_y, square_size)
        if region_3d is None:
            return None

        # Extract the median pixels and the starting indices of the square region
        median_x_full, median_y_full, start_x, start_y = median_pixels(
            aligned_data, center_x, center_y, square_size, percentage
        )

        # Convert absolute coordinates to relative coordinates within the region_3d array
        median_x_relative = median_x_full - start_x
        median_y_relative = median_y_full - start_y

        # Initialize rolling average for this file
        rolling_avg = None
        num_pixels = 0

        # Use ProcessPoolExecutor with limited workers to reduce CPU usage
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
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
                        rolling_avg = (
                            wavelength_data  # Initialize with the first spectrum
                        )
                    else:
                        rolling_avg = (rolling_avg * num_pixels + wavelength_data) / (
                            num_pixels + 1
                        )

                    # Increment the number of pixels processed
                    num_pixels += 1

        print(f"Final rolling average for {file_path}: {rolling_avg}")

        # Generate wavelengths (indices or actual wavelengths if available)
        wavelengths = np.arange(len(rolling_avg))  # Use indices as wavelengths
        return file_path, wavelengths, rolling_avg

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def wavelength_calibration(folder_path: str, square_size: int = 100, percentage: int = 1):
    """
    Performs wavelength calibration by aligning spectra from all files to the first file's spectrum.
    Saves all aligned files (including the reference) to the "Wavelength" folder with "_wave" suffix.
    """
    spatial_folder = os.path.join(folder_path, "AlignedImages", "Spatial")
    
    if not os.path.exists(spatial_folder):
        print(f"Spatial folder not found: {spatial_folder}")
        return

    spatial_files = [
        os.path.join(spatial_folder, f) 
        for f in os.listdir(spatial_folder) 
        if f.endswith(".fits")
    ]
    
    if not spatial_files:
        print("No files found in Spatial folder")
        return

    # Load existing shifts (if any)
    shifts_file = os.path.join(folder_path, "shifts.csv")
    shifts = load_shifts(shifts_file)

    # Process the first file to use as the reference spectrum
    ref_file = spatial_files[0]
    print(f"Using {ref_file} as the reference spectrum.")

    ref_result = process_file(ref_file, square_size, percentage, max_workers=4)
    if ref_result is None:
        print(f"Failed to process reference file {ref_file}. Exiting.")
        return

    ref_file_name, ref_wavelengths, ref_intensities = ref_result

    # Save the reference spectrum to the Wavelength folder with "_wave" suffix
    save_aligned_spectrum(ref_file, 0, ref_wavelengths)  # Shift=0 for reference

    # Store the reference spectrum
    original_spectra = [(ref_file_name, ref_wavelengths, ref_intensities)]
    aligned_spectra = [(ref_file_name, ref_wavelengths, ref_intensities)]

    # Process the remaining files
    for file in spatial_files[1:]:
        result = process_file(file, square_size, percentage, max_workers=4)
        if result is not None:
            file_name, target_wavelengths, target_intensities = result

            # Save original spectrum
            original_spectra.append((file_name, target_wavelengths, target_intensities))

            # Check if the file already has a wavelength shift in the shifts dictionary
            if file in shifts:
                shift_y, shift_x, wavelength_shift = shifts[file]
                print(f"Using existing wavelength shift for {file}: {wavelength_shift}")
            else:
                # Calculate the shift required to align the target spectrum to the reference spectrum
                wavelength_shift = calculate_spectral_shift(ref_intensities, target_intensities)
                print(f"Calculated wavelength shift for {file}: {wavelength_shift}")

                # Update the shifts dictionary
                if file in shifts:
                    shift_y, shift_x = shifts[file]
                else:
                    shift_y, shift_x = 0, 0  # Default spatial shifts if not found
                shifts[file] = (shift_y, shift_x, wavelength_shift)

            # Align the target spectrum to the reference spectrum
            aligned_intensities = align_spectrum(ref_wavelengths, ref_intensities, target_wavelengths, target_intensities)
            aligned_spectra.append((file_name, ref_wavelengths, aligned_intensities))

            # Save the aligned spectrum to the Wavelength folder
            save_aligned_spectrum(file, wavelength_shift, ref_wavelengths)

    # Save the updated shifts to the CSV file
    save_shifts(shifts_file, shifts)


def align_spectrum(
    ref_wavelengths, ref_intensities, target_wavelengths, target_intensities
):
    """
    Aligns the target spectrum to the reference spectrum using cross-correlation and interpolation.
    """

    # Calculate the shift required to align the target spectrum to the reference spectrum
    shift = calculate_spectral_shift(ref_intensities, target_intensities)

    if shift == 0:
        print("Spectra are already aligned. No shift applied.")
        return target_intensities  # Return the original target intensities

    # Shift the target spectrum
    shifted_intensities = np.roll(target_intensities, shift)

    # Create an interpolation function for the shifted target spectrum
    interpolate_func = interp1d(
        target_wavelengths, shifted_intensities, kind="linear", fill_value="extrapolate"
    )

    # Interpolate shifted intensities onto the reference wavelength grid
    aligned_intensities = interpolate_func(ref_wavelengths)

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
    subset_size = min(
        len(ref_normalized), len(target_normalized)
    )  # Use the full spectrum
    ref_subset = ref_normalized[:subset_size]
    target_subset = target_normalized[:subset_size]

    # Calculate cross-correlation
    correlation = correlate(ref_subset, target_subset, mode="full")
    shift = np.argmax(correlation) - (len(ref_subset) - 1)

    print(f"Cross-correlation result: {correlation}")  # Debug print
    print(f"Calculated shift: {shift}")  # Debug print
    return shift


def plot_spectra(spectra, title, save_path):
    plt.figure(figsize=(12, 6))
    for file, wavelengths, intensities in spectra:
        label = os.path.basename(file)  # Show only filename, not full path
        plt.plot(wavelengths, intensities, label=label)
    plt.xlabel("Wavelength (pixel index)")
    plt.ylabel("Intensity (ADU)")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # Avoid legend overlap
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {save_path}")


def intensity_calibration(folder_path, square_size=100, percentage=1):
    """Calibrate intensity and save reference to Intensity folder with _scaled suffix"""
    wavelength_folder = os.path.join(folder_path, "AlignedImages", "Wavelength")
    files = [
        os.path.join(wavelength_folder, f) 
        for f in os.listdir(wavelength_folder) 
        if f.endswith("_wave.fits")
    ]
    
    if not files:
        print("No wavelength-aligned files found")
        return

    # Process reference spectrum
    ref_file = files[0]
    ref_result = process_file(ref_file, square_size, percentage, max_workers=4)
    if not ref_result:
        print("Failed to process reference file")
        return
    
    ref_name, ref_wavelengths, ref_intensities = ref_result
    ref_mask = (ref_intensities != 0)

    # Save reference to Intensity folder (scale factor = 1.0)
    save_scaled_spectrum(ref_file, ref_intensities, ref_wavelengths)
    print(f"Saved reference spectrum to Intensity folder: {os.path.basename(ref_file).replace('.fits', '_scaled.fits')}")

    # Store spectra for plotting
    original_spectra = [(ref_name, ref_wavelengths, ref_intensities)]
    calibrated_spectra = [(ref_name, ref_wavelengths, ref_intensities)]

    # Process target files
    for file in files[1:]:
        result = process_file(file, square_size, percentage, max_workers=4)
        if not result:
            print(f"Skipping {file} - processing failed")
            continue

        file_name, target_wavelengths, target_intensities = result
        original_spectra.append((file_name, target_wavelengths, target_intensities))
        
        # Create combined mask
        target_mask = (target_intensities != 0)
        valid_mask = ref_mask & target_mask

        if not np.any(valid_mask):
            print(f"Warning: No valid wavelength overlap for {file_name}")
            scaling_factor = 1.0
        else:
            ref_integral = np.trapezoid(ref_intensities[valid_mask], ref_wavelengths[valid_mask])
            target_integral = np.trapezoid(target_intensities[valid_mask], target_wavelengths[valid_mask])
            scaling_factor = ref_integral / target_integral

        scaled_intensities = target_intensities * scaling_factor
        calibrated_spectra.append((file_name, ref_wavelengths, scaled_intensities))
        save_scaled_spectrum(file, scaled_intensities, ref_wavelengths)

    # Generate plots
    plot_spectra(original_spectra, "Original Spectra", 
                os.path.join(folder_path, "original_spectra.png"))
    plot_spectra(calibrated_spectra, "Calibrated Spectra", 
                os.path.join(folder_path, "calibrated_spectra.png"))

## LOADING FUNCTIONS


def load_shifts(file_path: str) -> dict:
    """
    Load shifts including circle centers.
    Returns: {filename: (shift_y, shift_x, wavelength_shift, cx, cy)}
    """
    shifts = {}
    try:
        if os.path.exists(file_path):
            with open(file_path, mode="r") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if not row:  # Skip empty rows
                        continue
                        
                    filename = row[0]
                    try:
                        # Safely extract values with defaults
                        shift_y = float(row[1]) if len(row) > 1 else 0.0
                        shift_x = float(row[2]) if len(row) > 2 else 0.0
                        wavelength_shift = int(row[3]) if len(row) > 3 else 0
                        cx = float(row[4]) if len(row) > 4 else 0.0
                        cy = float(row[5]) if len(row) > 5 else 0.0
                        
                        shifts[filename] = (shift_y, shift_x, wavelength_shift, cx, cy)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Malformed row for {filename} - {str(e)}")
                        shifts[filename] = (0.0, 0.0, 0, 0.0, 0.0)  # Default values
                        
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

    with fits.open(filename, menmap=True) as hdul:
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


def load_3d_region(
    file_path: str, center_x: int, center_y: int, square_size: int = 100
):
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


## SAVING FUNCTIONS


def save_shifts(file_path: str, shifts: dict):
    """Save shifts with all 5 values (shift_y, shift_x, wavelength_shift, cx, cy)"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "shift_y", "shift_x", "wavelength_shift", "cx", "cy"])
            
            for filename, values in shifts.items():
                # Ensure we have all 5 values
                full_values = list(values) + [0]*(5-len(values))
                writer.writerow([filename] + full_values[:5])
                
    except Exception as e:
        print(f"Error saving shifts: {str(e)}")


def save_aligned_fits(
    original_path: str, new_folder: str, aligned_data, shift_x: float, shift_y: float
):
    """
    Save the spatially aligned data as a new FITS file in the specified folder.

    Args:
        original_path (str): Path to the original FITS file.
        new_folder (str): Folder to save the new aligned FITS file (e.g., "Spatial").
        aligned_data: The aligned image data.
        shift_x (float): The x-axis shift applied.
        shift_y (float): The y-axis shift applied.
    """
    # Create the new file path
    path, file = os.path.split(original_path)
    aligned_folder = os.path.join(path, "AlignedImages", new_folder)
    os.makedirs(aligned_folder, exist_ok=True)
    new_file_name = f"{os.path.splitext(os.path.basename(original_path))[0]}_aligned.fits"
    new_file_path = os.path.join(aligned_folder, new_file_name)

    # Debug print
    print(f"Saving aligned file to: {new_file_path}")

    # Read the original FITS file
    with fits.open(original_path) as hdul:
        # Update the header with the shift information
        hdul[0].header["SHIFT_X"] = (shift_x, "X-axis shift applied during alignment")
        hdul[0].header["SHIFT_Y"] = (shift_y, "Y-axis shift applied during alignment")

        # Replace the data with the aligned data
        hdul[0].data = aligned_data

        # Save the new FITS file
        hdul.writeto(new_file_path, overwrite=True)

    # Debug print
    print(f"File saved successfully: {new_file_path}")


def save_aligned_spectrum(file_path: str, shift: int, ref_wavelengths: np.ndarray):
    """
    Saves the aligned spectrum to a new .fits file in the "Wavelength" folder.

    Args:
        file_path (str): Path to the original .fits file.
        shift (int): The calculated shift value (e.g., -3).
        ref_wavelengths (np.ndarray): Wavelengths of the reference spectrum (1D array).
    """
    # Create the new file path
    aligned_folder = os.path.join(os.path.dirname(os.path.dirname(file_path)), "Wavelength")
    os.makedirs(aligned_folder, exist_ok=True)
    new_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_wave.fits"
    new_file_path = os.path.join(aligned_folder, new_file_name)

    with fits.open(file_path) as hdul:
        # Ensure the original data is 3D
        if hdul[1].data.ndim != 3:
            raise ValueError(
                "The original data must be 3D (expected [wavelength, x, y])."
            )

        # Create a copy of the original data to avoid modifying it directly
        new_data = hdul[1].data.copy()

        # Apply the shift to all pixels
        new_data = np.roll(new_data, shift=shift, axis=0)  # Shift along the wavelength axis

        # Replace the wrapped wavelengths with zeros
        if shift < 0:
            # If the shift is negative, the wrapped values are at the end
            new_data[shift:, :, :] = 0  # Set the last `shift` wavelengths to zero
        elif shift > 0:
            # If the shift is positive, the wrapped values are at the beginning
            new_data[:shift, :, :] = 0  # Set the first `shift` wavelengths to zero

        # Update the data in the HDU
        hdul[1].data = new_data

        # Update the header with the new wavelength information
        hdul[1].header["WAVE"] = (
            str(ref_wavelengths.tolist()),
            "Reference wavelengths",
        )
        hdul[1].header["SHIFT"] = (shift, "Wavelength shift applied during alignment")

        # Save the new .fits file
        hdul.writeto(new_file_path, overwrite=True)


def save_scaled_spectrum(file_path, scaled_intensities, wavelengths):
    """
    Save the scaled spectrum to the "Intensity" folder with "_scaled.fits" suffix.
    """
    # Extract the base filename (e.g., "RSM20230207T001_FE_aligned_wave.fits")
    original_filename = os.path.basename(file_path)
    scaled_filename = original_filename.replace(".fits", "_scaled.fits")

    # Create the output folder path
    intensity_folder = os.path.join(
        os.path.dirname(os.path.dirname(file_path)),  # Go up from Wavelength/Spatial
        "Intensity"
    )
    os.makedirs(intensity_folder, exist_ok=True)

    # Full output path
    output_path = os.path.join(intensity_folder, scaled_filename)

    # Save as new FITS file
    with fits.open(file_path) as hdul:
        hdul[1].data = scaled_intensities  # Replace data with scaled intensities
        hdul[1].header["SCALED"] = (True, "Intensity-scaled spectrum")
        hdul.writeto(output_path, overwrite=True)
        print(f"Saved scaled spectrum to: {output_path}")

## Folder Structure Logic

def ensure_folder_structure(folder_path: str):
    """
    Ensures the required folder structure exists:
    - TestImages/AlignedImages/Spatial
    - TestImages/AlignedImages/Wavelength
    - TestImages/AlignedImages/Intensity

    Args:
        folder_path (str): Path to the main folder (e.g., "TestImages").
    """
    aligned_folder = os.path.join(folder_path, "AlignedImages")
    spatial_folder = os.path.join(aligned_folder, "Spatial")
    wavelength_folder = os.path.join(aligned_folder, "Wavelength")
    intensity_folder = os.path.join(aligned_folder, "Intensity")

    # Create folders if they don't exist
    os.makedirs(spatial_folder, exist_ok=True)
    os.makedirs(wavelength_folder, exist_ok=True)
    os.makedirs(intensity_folder, exist_ok=True)

def find_files_in_order(folder_path: str, file_extension: str, search_order: list):
    files = []
    for folder_name in search_order:
        search_folder = os.path.join(folder_path, "AlignedImages", folder_name)
        if os.path.isdir(search_folder):
            print(f"Searching in folder: {search_folder}")
            folder_files = [
                os.path.abspath(os.path.join(search_folder, f))  # Return absolute path
                for f in os.listdir(search_folder)
                if f.lower().endswith(file_extension)
            ]
            print(f"Found files: {folder_files}")
            files.extend(folder_files)
    return files


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
            wavelength_calibration("TestImages", 100, 1)
            intensity_calibration("TestImages", 100, 1)

    except Exception as e:
        print(f"Critical error: {str(e)}")