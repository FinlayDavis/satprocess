# BSD 3-Clause License
#
# Copyright (c) [2025], [Finlay Davis]
# All rights reserved.

import os
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.ndimage as sp
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_otsu
from scipy.interpolate import interp1d
from scipy.signal import correlate
from typing import Dict, Tuple, Optional, List

# Setting up plotting font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Setting logger at the "INFO" level
# Possible logging levels are "NOTSET", "INFO", "WARN", "ERROR" and "CRITICAL" - depending on use-case
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Shift Management Functions
def clean_filename(filename: str) -> str:
    """Function to remove appendages onto filenames, so that comparison accross calibrated files can be made.

    Args:
        filename (str): The filename to be cleaned.

    Returns:
        str: The cleaned filename
    """
    suffixes = ['_aligned', '_wave', '_scaled', '_spatial']
    for suffix in suffixes:
        filename = filename.replace(suffix, '')
    return os.path.splitext(filename)[0] + '.fits'

def load_shifts(shifts_file: str) -> Dict[str, Tuple]:
    """Load in any known calibration values into a dictionary, to be modified throughout the code.

    Args:
        shifts_file (str): The name of the file to load.

    Returns:
        Dict[str, Tuple]: Returns a dict which associates the filename with the calibrated values.
    """
    shifts = {}
    try:
        if os.path.exists(shifts_file):
            with open(shifts_file, mode="r") as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    if not row:
                        continue
                    filename = clean_filename(row[0])
                    try:
                        values = [
                            float(row[1]) if len(row) > 1 else 0.0,   # shift_y
                            float(row[2]) if len(row) > 2 else 0.0,   # shift_x
                            int(row[3]) if len(row) > 3 else 0,       # wavelength_shift
                            float(row[4]) if len(row) > 4 else 0.0,   # cx
                            float(row[5]) if len(row) > 5 else 0.0,   # cy
                            float(row[6]) if len(row) > 6 else 1.0    # intensity_scaling
                        ]
                        shifts[filename] = tuple(values)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Malformed row for {filename}: {str(e)}")
                        shifts[filename] = (0.0, 0.0, 0, 0.0, 0.0, 1.0)
    except Exception as e:
        logger.error(f"Error loading shifts: {str(e)}")
    return shifts

def save_shifts(shifts: Dict[str, Tuple], shifts_file: str):
    """Save the calibration values to the shifts.csv file.

    Args:
        shifts (Dict[str, Tuple]): The updated dict, with new calibration values.
        shifts_file (str): The name of the file to be written to.
    """
    try:
        os.makedirs(os.path.dirname(shifts_file), exist_ok=True)
        with open(shifts_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "filename", "shift_y", "shift_x", "wavelength_shift", 
                "cx", "cy", "intensity_scaling"
            ])
        
            for filename in sorted(shifts.keys()):
                values = shifts[filename]
                writer.writerow([filename] + list(values))
    except Exception as e:
        logger.error(f"Error saving shifts: {str(e)}")

def get_shifts(shifts: Dict[str, Tuple], filename: str) -> Tuple:
    """Outputs the calibration values for a specific file, independent of the appended names.

    Args:
        shifts (Dict[str, Tuple]): Full calibration shift dictionary.
        filename (str): The name of the file for which the calibration values are required.

    Returns:
        Tuple: The current calibration values associated with that filename.
    """
    clean_name = clean_filename(filename)
    return shifts.get(clean_name, (0.0, 0.0, 0, 0.0, 0.0, 1.0))

def update_shifts(shifts: Dict[str, Tuple], filename: str, **updates):
    """Update calibration values in the shifts dictionary, depending on which values have been changed.

    Args:
        shifts (Dict[str, Tuple]): Full calibration shift dictionary.
        filename (str): The name of the file which will have the calibration values changed.
    """
    clean_name = clean_filename(filename)
    current = list(get_shifts(shifts, clean_name))
    fields = ['shift_y', 'shift_x', 'wavelength_shift', 'cx', 'cy', 'intensity_scaling']
    
    for field, value in updates.items():
        if field in fields:
            idx = fields.index(field)
            current[idx] = value
    
    shifts[clean_name] = tuple(current)

# FITS Processing Functions
def ensure_folder_structure(base_path: str, H_alpha: bool = False):
    """Creates the appropriate folder structure the code needs, if it doesn't exist already.

    Args:
        base_path (str): The parent directory where the subfolders are created.
        H_alpha (bool): Whether there will need to be a specific H_alpha folder created. Defaults to "False".
    """
    if H_alpha:
        folders = ["Spatial", "Wavelength", "Intensity", "H_alpha"]
    else:
        folders = ["Spatial", "Wavelength", "Intensity"]
    for folder in folders:
        os.makedirs(os.path.join(base_path, "AlignedImages", folder), exist_ok=True)

def load_fits(file_path: str, mode: str = "full", **kwargs) -> np.ndarray:
    """Loads FITS data with memory-efficient spectrum extraction.
    
    Supports multiple loading modes for different dimensionalities and regions of interest.
    
    Args:
        file_path (str): Path to the FITS file to load
        mode (str): Loading mode:
            - "full": Load entire data array (default)
            - "spectrum": Extract spectrum at given position(s)
            - "slice": Extract single wavelength layer
            - "region": Extract 3D sub-region around center
        **kwargs: Additional mode-specific arguments:
            For "spectrum":
                - wavelength (int): Wavelength index to extract
                - x (int): X position (None for full column)
                - y (int): Y position (None for full row)
            For "slice":
                - wavelength (int): Wavelength index to extract
            For "region":
                - center_x (int): X center position
                - center_y (int): Y center position
                - size (int): Region size (default: 100)
    
    Raises:
        FileNotFoundError: If specified file doesn't exist
        ValueError: For invalid modes or missing required arguments
    
    Returns:
        np.ndarray: Requested data array with appropriate dimensionality
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    with fits.open(file_path, memmap=True, do_not_scale_image_data=True) as hdul:
        if len(hdul) < 2:
            raise ValueError("FITS file missing data extension")
        
        # Handle 1D spectra
        if hdul[1].data.ndim == 1:
            return hdul[1].data.copy()

        # Handle 3D cubes
        elif hdul[1].data.ndim == 3:
            if mode == "spectrum":
                wavelength = kwargs.get('wavelength')
                x = kwargs.get('x')
                y = kwargs.get('y')

                if x is not None and y is not None:
                    # Extract single-pixel spectrum (shape: [wavelengths])
                    return hdul[1].section[:, y, x].astype(np.float32)
                elif y is not None:
                    # Extract column spectrum (shape: [wavelengths, x])
                    return hdul[1].section[wavelength, y, :].astype(np.float32)
                elif x is not None:
                    return hdul[1].section[wavelength, :, x].astype(np.float32)
                else:
                    raise ValueError("For spectrum mode, specify x, y, or both")
            
            elif mode == "slice":
                # Extract single wavelength layer (shape: [y, x])
                return hdul[1].section[kwargs['wavelength'], :, :].astype(np.float32)
            
            elif mode == "region":
                # Extract a 3D sub-region
                cx = kwargs.get('center_x')
                cy = kwargs.get('center_y')
                size = kwargs.get('size', 100)
                
                if cx is None or cy is None:
                    raise ValueError("For region mode, specify center_x and center_y")
                
                half = size // 2
                x_start = max(0, cx - half)
                x_end = min(hdul[1].data.shape[2], cx + half)
                y_start = max(0, cy - half)
                y_end = min(hdul[1].data.shape[1], cy + half)
                
                return hdul[1].section[:, y_start:y_end, x_start:x_end].astype(np.float32)
            
            
            elif mode == "full":
                return hdul[1].data.astype(np.float32)
            
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            raise ValueError(f"Unsupported data dimensionality: {hdul[1].data.ndim}")

def save_processed(
    original_path: str,
    base_path: str,
    output_type: str,
    data: np.ndarray,
    suffix: str = "",
    metadata: Optional[Dict] = None
) -> str:
    """
    Save processed data to appropriate subfolder with metadata
    
    Args:
        original_path (str): Path to original file
        base_path (str): Base directory path
        output_type (str): One of 'spatial', 'wavelength', 'intensity'
        data (np.ndarray): Processed data to save
        suffix (str): Filename suffix (default: "")
        metadata (Dict): Dictionary of header updates (optional) (default: None) 
    
    Returns:
        str: Path to saved file
    """
    output_folder = os.path.join(base_path, "AlignedImages", output_type.capitalize())
    os.makedirs(output_folder, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}{suffix}.fits")

    # In save_processed, before saving:
    logger.debug(f"Saving data with shape: {data.shape}")
    with fits.open(original_path) as hdul:
        # Update headers
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, tuple):
                    hdul[0].header[key] = value[0], value[1]
                else:
                    hdul[0].header[key] = value
        
        # Update data
        hdul[1].data = data
        hdul.writeto(output_path, overwrite=True)
    
    logger.info(f"Saved {output_type} data to {output_path}")
    return output_path

def find_matching_ha_file(fe_file: str) -> Optional[str]:
    """Find matching H-alpha file for a given Fe file.
    
    Args:
        fe_file (str): Path to the Fe FITS file
    
    Returns:
        str: Path to matching H-alpha file if found, None otherwise
    """

    base_name = os.path.basename(fe_file)
    if '_FE.fits' in base_name:
        ha_file = base_name.replace('_FE.fits', '_HA.fits')
        return os.path.join(os.path.dirname(fe_file), ha_file)
    return None

# Image Processing Functions
def preprocess_image(image: np.ndarray, sigma: float = 2) -> np.ndarray:
    """Prepare image for circle detection through edge enhancement.
    
    Applies Gaussian blur, Otsu thresholding, and Canny edge detection.
    
    Args:
        image (np.ndarray): Input image array
        sigma (float): Standard deviation for Gaussian blur
    
    Returns:
        np.ndarray: Binary edge detection result
    """
    
    blurred = sp.gaussian_filter(image, sigma=sigma)
    thresh = threshold_otsu(blurred)
    binary = blurred > thresh
    return canny(binary, sigma=1)

def detect_circles(
    edges: np.ndarray, 
    min_radius: Optional[int] = None, 
    max_radius: Optional[int] = None,
    reference_radius: Optional[int] = None,
    margin_percent: float = 2.0
) -> np.ndarray:
    """Detect circles using Hough transform with an adaptive radius range.
    
    Args:
        edges (np.ndarray): Edge detection result
        min_radius (int): Minimum circle radius (optional)
        max_radius (int): Maximum circle radius (optional)
        reference_radius: Previously detected radius for adaptive range
        margin_percent: Percentage margin around reference radius
        
    Returns:
        Array of detected circles (cx, cy, radius)
    """

    # Use adaptive range if reference radius is provided
    if reference_radius is not None:
        margin = reference_radius * margin_percent / 100
        min_radius = int(reference_radius - margin)
        max_radius = int(reference_radius + margin)
    else:
        # Default wide range if no reference
        min_radius = min_radius or 500
        max_radius = max_radius or 1500
    
    radii = np.arange(min_radius, max_radius, 1)
    hough_res = hough_circle(edges, radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, radii, total_num_peaks=1)
    
    return np.array(list(zip(cx, cy, radii)))

### REMOVE FUNCTION
def transform_image(image: np.ndarray, dx: float, dy: float, mode: str = "nearest") -> np.ndarray:
    """Shift image by specified amount"""
    return sp.shift(image, shift=[dy, dx], mode=mode)

def extract_median_pixels(
    image: np.ndarray,
    x_coord: int,
    y_coord: int,
    size: int = 100,
    percentage: float = 1
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Extract median percentage of pixels from square region around the inputted coordinates.
    
    Args:
        image (np.ndarray): Input image array
        x_coord (int): X coordinate of region center
        y_coord (int): Y coordinate of region center
        size (int): Width/height of square region (default: 100)
        percentage (float): Percentage of median pixels to extract (default: 1%)
    
    Returns:
        Tuple containing:
        - np.ndarray: x coordinates of selected pixels (absolute)
        - np.ndarray: y coordinates of selected pixels (absolute)
        - int: x start coordinate of region
        - int: y start coordinate of region
    """
    
    half = size // 2
    y_start, y_end = max(0, y_coord - half), min(image.shape[0], y_coord + half)
    x_start, x_end = max(0, x_coord - half), min(image.shape[1], x_coord + half)
    
    region = image[y_start:y_end, x_start:x_end]
    flat = region.flatten()
    sorted_idx = np.argsort(flat)
    
    med_start = int(len(sorted_idx) * (50 - percentage/2) / 100)
    med_end = int(len(sorted_idx) * (50 + percentage/2) / 100)
    median_idx = sorted_idx[med_start:med_end]
    
    y, x = np.unravel_index(median_idx, region.shape)
    return x + x_start, y + y_start, x_start, y_start

# Spectrum Processing Functions
def calculate_shift(ref_spec: np.ndarray, target_spec: np.ndarray) -> int:
    """Calculate optimal shift between reference and target spectra using cross-correlation.
    
    Args:
        ref_spec (np.ndarray): Reference spectrum array
        target_spec (np.ndarray): Target spectrum array to align
    
    Returns:
        int: Optimal shift amount (in pixels) to align target to reference
    """

    ref_norm = (ref_spec - np.mean(ref_spec)) / np.std(ref_spec)
    target_norm = (target_spec - np.mean(target_spec)) / np.std(target_spec)
    
    size = min(len(ref_norm), len(target_norm))
    correlation = correlate(ref_norm[:size], target_norm[:size], mode="full")
    return np.argmax(correlation) - (size - 1)

def align_spectrum(
    ref_wavelengths: np.ndarray,
    ref_intensities: np.ndarray,
    target_wavelengths: np.ndarray,
    target_intensities: np.ndarray
) -> np.ndarray:
    """Align target spectrum to reference spectrum.
    
    Args:
        ref_wavelengths (np.ndarray): Wavelength array of reference
        ref_intensities (np.ndarray): Intensity array of reference
        target_wavelengths (np.ndarray): Wavelength array of target
        target_intensities (np.ndarray): Intensity array of target
    
    Returns:
        np.ndarray: Interpolated target intensities aligned to reference wavelengths
    """
    
    shift = calculate_shift(ref_intensities, target_intensities)
    shifted = np.roll(target_intensities, shift)
    
    if shift < 0:
        shifted[shift:] = 0
    elif shift > 0:
        shifted[:shift] = 0
    
    interp_func = interp1d(target_wavelengths, shifted, kind='linear', fill_value='extrapolate')
    return interp_func(ref_wavelengths)

def calculate_scaling(ref_spec: np.ndarray, target_spec: np.ndarray) -> float:
    """Calculate intensity scaling factor between reference and target spectra.
    
    Computes ratio of integrated fluxes between non-zero regions.
    
    Args:
        ref_spec (np.ndarray): Reference spectrum array
        target_spec (np.ndarray): Target spectrum array to scale
    
    Raises:
        ValueError: If no valid overlapping non-zero regions found
    
    Returns:
        float: Scaling factor to apply to target spectrum
    """

    mask = (ref_spec != 0) & (target_spec != 0)
    if not np.any(mask):
        return 1.0
    
    ref_integral = np.trapezoid(ref_spec[mask])
    target_integral = np.trapezoid(target_spec[mask])
    return ref_integral / target_integral if target_integral != 0 else 1.0

def plot_spectra(
    spectra_data: List[Tuple[str, np.ndarray, np.ndarray]], 
    title: str,
    save_path: str,
    figsize: Tuple[int, int] = (12, 6),
    normalize: bool = False
):
    """
    Plot multiple spectra on the same figure with optional normalization
    
    Args:
        spectra_data (List(Tuple(str, np.ndarray, np.ndarray))): List of (filename, wavelengths, intensities) tuples
        title (str): Plot title
        save_path (str): Where to save the plot
        figsize (Tuple(int, int)): Figure dimensions (default: 12, 6)
        normalize (bool): Whether to normalize spectra to [0,1] range (default: False)
    """
    plt.figure(figsize=figsize)
    
    for file_path, wavelengths, intensities in spectra_data:
        # Handle multi-dimensional data
        if intensities.ndim > 1:
            intensities = intensities.mean(axis=tuple(range(1, intensities.ndim)))
        
        # Optional normalization
        if normalize:
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        
        label = os.path.basename(file_path)
        plt.plot(wavelengths, intensities, label=label)
    
    plt.xlabel("Wavelength (pixel index)")
    plt.ylabel("Intensity (ADU)")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved spectrum plot to {save_path}")

def display_slice(
    file_path: str,
    wavelength: int,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = None,
    cmap: str = 'viridis',
    show: bool = True
) -> plt.Figure:
    """
    Display a single wavelength slice from a FITS file as a figure.
    
    Args:
        file_path (str): Path to the FITS file
        wavelength (int): Wavelength index to display
        save_path (Optional[str]): Path to save the figure (default: None)
        figsize (Tuple[int, int]): Figure size (default: (10, 6))
        title (Optional[str]): Figure title (default: None)
        cmap (str): Colormap for the image (default: 'viridis')
        show (bool): Whether to display the figure (default: True)
    
    Returns:
        plt.Figure: The created figure object
    """
    try:
        # Load the slice
        image_data = load_fits(file_path, mode="slice", wavelength=wavelength)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Display the image
        plt.imshow(image_data, cmap=cmap, origin='lower')
        plt.colorbar(label='Intensity (ADU)')
        
        # Set title
        if title is None:
            title = f"Wavelength Slice {wavelength} from {os.path.basename(file_path)}"
        plt.title(title)
        
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved slice plot to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to plot slice from {file_path}: {str(e)}")
        raise


def get_central_coords(file_path, shifts):
    """ TODO update summary

    Args:
        file_path (str): The path to the file which will have the central coordinates extracted.
        shifts (Dict(str, Tuple)):  Dictionary of calibration parameters tracking:
               (shift_y, shift_x, wavelength_shift, cx, cy, intensity_scaling)

    Returns:
        cx, cy (int): The central coordinates of the file privided
    """

    base_name = os.path.splitext(os.path.basename(file_path))[0] + '.fits'
    current_shifts = get_shifts(shifts, base_name)  
    
    if current_shifts[3:5] != (0.0, 0.0):
        cx, cy = current_shifts[3:5]
    else:
        alignment_slice = load_fits(file_path, mode="slice", wavelength=1)
        edges = preprocess_image(alignment_slice)
        circles = detect_circles(edges)
        if circles.size == 0:
            logger.warning(f"No circles detected in {file_path}")
            return None
        cx, cy, _ = circles[0]
        update_shifts(shifts, base_name, cx=cx, cy=cy)
    return int(cx), int(cy)


# Pipeline Functions
def process_spectrum(
    file_path: str,
    x_coord: int,
    y_coord: int,
    region_size: int,
    percentage: float,
    calibration_wavelength: int = 10
) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """Process a spectral data file to extract representative spectrum.
    
    TODO fix description
    Handles both pre-aligned 1D spectra and 3D spectral cubes. For 3D data:
    - Locates center coordinates (either from stored shifts or via circle detection)
    - Extracts a region around the center
    - Selects median pixels within the region
    - Computes average spectrum from selected pixels

    Args:
        file_path (str): Path to the FITS file to process. If filename contains '_wave.fits',
                  treats as pre-aligned 1D spectrum.
        TODO update with x and y coords info
        region_size (int): Size (in pixels) of the square region to extract around center
        percentage (float): Percentage of median pixels to select within the region (0-100)

    Returns:
        Tuple containing:
        - str: Original file path
        - np.ndarray: Wavelength array (pixel indices)
        - np.ndarray: Extracted spectrum array
        Returns None if processing fails or no circle detected in 3D data

    Notes:
        - Updates shifts dictionary with new center coordinates if detected
        - For wavelength-aligned files, simply returns the 1D spectrum
        - Handles both Fe and H-alpha files based on input path
    """

    try:
        # Load 3D region
        region_3d = load_fits(
            file_path,
            mode="region",
            center_x=x_coord,
            center_y=y_coord,
            size=region_size
        )
        
        # Get median pixels
        wave = 10
        logger.debug(type(wave))
        logger.debug(wave)
        alignment_slice = load_fits(file_path, mode="slice", wavelength=wave)
        median_x, median_y, start_x, start_y = extract_median_pixels(
            alignment_slice, x_coord, y_coord, region_size, percentage
        )
        
        # Convert to relative coordinates
        median_x_rel = median_x - start_x
        median_y_rel = median_y - start_y
        
        # Calculate average spectrum
        spectra = []
        for x, y in zip(median_x_rel, median_y_rel):
            spectrum = region_3d[:, y, x]
            spectra.append(spectrum)
        
        avg_spectrum = np.mean(spectra, axis=0)
        wavelengths = np.arange(len(avg_spectrum))
        
        return file_path, wavelengths, avg_spectrum
        
    except Exception as e:
        logger.error(f"Spectrum processing failed for {file_path}: {str(e)}")
        return None

def spatial_calibration(
    folder_path: str,
    shifts: Dict[str, Tuple],
    alignment_wavelength: int = 1,
    reference_index: int = 0,
    wide_min_radius: int = 500,
    wide_max_radius: int = 1500,
    tight_margin_percent: float = 2.0,
    mode: str = "fe_only",
    force_recalculate: bool = False
) -> List[str]:
    """Perform spatial alignment of spectral data cubes using circular features.
    
    Implements a two-stage adaptive circle detection algorithm:
    1. Wide search on reference file to establish initial parameters
    2. Tightly constrained search on subsequent files using reference parameters
    
    Creates spatially aligned versions of input files and updates calibration shifts.

    Args:
        folder_path (str): Directory containing FITS files to process
        shifts (Dict(str, Tuple)): Dictionary tracking calibration parameters
        alignment_wavelength (int): Wavelength layer to use for alignment (default: 1)
        reference_index (int): Index of file to use as reference (default: 0)
        wide_min_radius (int): Minimum radius for initial wide search (default: 500)
        wide_max_radius (int): Maximum radius for initial wide search (default: 1500)
        tight_margin_percent (float): Percentage margin around reference radius for 
                            constrained searches (default: 2.0)
        mode (str): File selection mode:
              - "fe_only": Process only Fe files (default)
              - "all_files": Process all FITS files
        force_recalculate (bool): If True, recalculates shifts even if they exist in shifts file (default: False)

    Returns:
        List[str]: List of paths to processed/aligned files

    Raises:
        ValueError: If no matching files found or invalid reference index
        RuntimeError: If reference radius cannot be established
    """
    logger.info(f"Starting adaptive spatial calibration in {mode} mode...")
    
    files = [f for f in os.listdir(folder_path) 
            if f.lower().endswith('_fe.fits' if mode == "fe_only" else '.fits')]
    
    if not files:
        raise ValueError("No FITS files found matching mode criteria")
    if reference_index >= len(files):
        raise ValueError("Invalid reference index")

    # Track reference radius for adaptive detection
    reference_radius = None
    aligned_files = []

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        try:
            base_name = os.path.splitext(file)[0] + '.fits'
            current_shifts = get_shifts(shifts, base_name)
            
            # Skip processing if shifts exist and recalculation = False
            if not force_recalculate and current_shifts[:3] != (0.0, 0.0, 0):
                logger.info(f"Skipping {file} - shifts already calculated")
                aligned_files.append(file_path)
                continue
                
            # Load alignment slice
            alignment_slice = load_fits(file_path, mode="slice", wavelength=alignment_wavelength)
            edges = preprocess_image(alignment_slice)
            
            if i == reference_index:
                logger.info(f"Performing initial wide search on reference file {file}")
                circles = detect_circles(edges, wide_min_radius, wide_max_radius)
                reference_radius = circles[0][2]
                logger.info(f"Reference radius found: {reference_radius} pixels")
            else:
                # Adaptive search for subsequent files
                if reference_radius is None:
                    raise RuntimeError("Reference radius not established")
                
                margin = reference_radius * tight_margin_percent / 100
                min_r = int(reference_radius - margin)
                max_r = int(reference_radius + margin)
                logger.debug(f"Using adaptive bounds: {min_r}-{max_r} pixels")
                circles = detect_circles(edges, min_r, max_r)
            
            cx, cy, radius = circles[0]
            
            # For non-reference files, calculate shifts relative to reference
            if i == reference_index:
                shift_x, shift_y = 0.0, 0.0
                ref_center = (cx, cy)
            else:
                shift_x = ref_center[0] - cx
                shift_y = ref_center[1] - cy
            
            update_shifts(shifts, base_name, cx=cx, cy=cy, shift_x=shift_x, shift_y=shift_y)
            
            # Apply shifts to full cube only if new shifts were calculated
            if force_recalculate or current_shifts[:3] == (0.0, 0.0, 0):
                full_cube = load_fits(file_path, mode="full")
                aligned_cube = np.zeros_like(full_cube)
                for j in range(full_cube.shape[0]):
                    aligned_cube[j] = sp.shift(full_cube[j], shift=[shift_y, shift_x], mode="nearest")
                
                # Save results
                saved_path = save_processed(
                    file_path,
                    folder_path,
                    "spatial",
                    aligned_cube,
                    "_aligned",
                    {
                        'SPATIAL': ('CALIBRATED', 'Spatial calibration applied'),
                        'SHIFT_X': (shift_x, 'X shift (pixels)'),
                        'SHIFT_Y': (shift_y, 'Y shift (pixels)'),
                        'DET_RAD': (radius, 'Detected circle radius')
                    }
                )
                aligned_files.append(saved_path)
            else:
                aligned_files.append(file_path)
            
            logger.info(f"Processed {file}: center=({cx:.1f},{cy:.1f}), radius={radius:.1f}")
            
        except Exception as e:
            logger.error(f"Failed to process {file}: {str(e)}")
            continue

    save_shifts(shifts, os.path.join(folder_path, "shifts.csv"))
    logger.info(f"Calibration complete. Reference radius: {reference_radius} pixels")
    return aligned_files

def wavelength_calibration(
    folder_path: str,
    shifts: Dict[str, Tuple],
    region_size: int = 100,
    percentage: float = 1,
    mode: str = "all_files"
) -> List[str]:
    """Perform wavelength calibration on spectral data cubes.
    
    Aligns spectra from multiple files to a reference spectrum using cross-correlation.
    
    Args:
        folder_path (str): Path containing spatially aligned files
        shifts (Dict(str, Tuple)): Dictionary tracking calibration parameters
        region_size (int): Size of region around center to analyze (default: 100)
        percentage (float): Percentage of median pixels to use (default: 1%)
        mode (str): Calibration mode ("fe_only" or "all_files") (default: all_files)
    
    Raises:
        ValueError: If no valid input files found
        RuntimeError: If reference processing fails
    
    Returns:
        List[str]: List of paths to processed wavelength-aligned files
    """

    logger.info(f"Starting wavelength calibration in {mode} mode...")
    
    spatial_folder = os.path.join(folder_path, "AlignedImages", "Spatial")
    if mode == "fe_only":
        files = [os.path.join(spatial_folder, f) for f in os.listdir(spatial_folder) 
                if f.endswith('_aligned.fits') and '_FE' in f]
    else:
        files = [os.path.join(spatial_folder, f) for f in os.listdir(spatial_folder) 
                if f.endswith('_aligned.fits')]

    if not files:
        raise ValueError("No spatially aligned files found")

    # Process reference
    ref_path = files[0]
    cx, cy = get_central_coords(ref_path, shifts)
    if "_wave.fits" in ref_path:
        ref_intensities = load_fits(ref_path, mode="spectrum", x=cx, y=cy)
        ref_waves = np.arange(len(ref_intensities))
    else:
        ref_result = process_spectrum(ref_path, cx, cy, region_size, percentage)
        if not ref_result:
            raise RuntimeError("Reference processing failed")
        ref_name, ref_waves, ref_intensities = ref_result

    ref_cube = load_fits(ref_path, mode="full")

    save_processed(
        ref_path,
        folder_path,
        "wavelength",
        ref_cube,
        "_wave",
        {'WAVE_REF': ('REFERENCE', 'Reference spectrum')}
    )

    aligned_files = [ref_path]
    for file in files[1:]:
        try:
            # Load the full 3D cube
            full_cube = load_fits(file, mode="full")
            cx, cy = get_central_coords(file, shifts)

            if "_wave.fits" in file:
                file_name = file
                target_waves = load_fits(file, mode="spectrum", x=cx, y=cy)
                target_intensities = np.arange(len(target_waves))
            else:
                target_result = process_spectrum(file, cx, cy, region_size, percentage)
                if not target_result:
                    logger.error(f"{file} processing failed")
                    continue
                else:
                    file_name, target_waves, target_intensities = target_result
                
            base_name = os.path.splitext(os.path.basename(file_name))[0] + '.fits'
            
            # Calculate or use existing wavelength shift
            current_shifts = get_shifts(shifts, base_name)
            if current_shifts[2] != 0:  # If wavelength shift exists
                shift = current_shifts[2]
            else:
                shift = calculate_shift(ref_intensities, target_intensities)
                update_shifts(shifts, base_name, wavelength_shift=shift)
            
            # Apply shift to full cube
            shifted_cube = np.roll(full_cube, shift, axis=0)
            if shift < 0:
                shifted_cube[shift:] = 0
            elif shift > 0:
                shifted_cube[:shift] = 0

            # Verify dimensions
            if shifted_cube.ndim != 3:
                logger.error(f"Unexpected dimensions in shifted cube: {shifted_cube.ndim}")
                continue
            
            # Save the full shifted cube
            saved_path = save_processed(
                file,
                folder_path,
                "wavelength",
                shifted_cube,
                "_wave",
                {
                    'WAVESHFT': (shift, 'Wavelength shift'),
                    'WAVEREF': ('ALIGNED', 'Aligned to reference')
                }
            )
            aligned_files.append(saved_path)
            
        except Exception as e:
            logger.error(f"Failed to process {file}: {str(e)}")
            continue

    save_shifts(shifts, os.path.join(folder_path, "shifts.csv"))

    return aligned_files

def intensity_calibration(
    folder_path: str,
    shifts: Dict[str, Tuple],
    region_size: int = 100,
    percentage: float = 1,
    mode: str = "all_files"
) -> List[str]:
    """Perform intensity calibration on spectral data cubes.
    
    Scales spectra from multiple files to match reference spectrum flux.
    
    Args:
        folder_path (str): Path containing wavelength-aligned files
        shifts (Dict(str, tuple)): Dictionary tracking calibration parameters
        region_size (int): Size of region around center to analyze (default: 100)
        percentage (float): Percentage of median pixels to use (default: 1%)
        mode (str): Calibration mode ("fe_only" or "all_files") (default: "all_files")
    
    Raises:
        ValueError: If ther's an incorrect calibration mode input or if no valid input files found
        RuntimeError: If reference processing fails
    
    Returns:
        List of paths to processed intensity-scaled files
    """
    
    wavelength_folder = os.path.join(folder_path, "AlignedImages", "Wavelength")
    if mode == "fe_only":
        files = [os.path.join(wavelength_folder, f) for f in os.listdir(wavelength_folder) 
                if f.endswith('_wave.fits') and '_FE' in f]
    elif mode == "all_files":
        files = [os.path.join(wavelength_folder, f) for f in os.listdir(wavelength_folder) 
                if f.endswith('_wave.fits')]
    else:
        raise ValueError('Incorrect mode inputted, use "fe_only" or "all_files"')
    
    logger.info(f"Starting intensity calibration in {mode} mode...")

    if not files:
        raise ValueError("No wavelength-aligned files found")

    # Process reference
    ref_path = files[0]
    cx, cy = get_central_coords(ref_path, shifts)
    if "_scaled.fits" in ref_path:
        ref_intensities = load_fits(ref_path, mode="spectrum", x=cx, y=cy)
        ref_waves = np.arange(len(ref_intensities))
    else:
        ref_result = process_spectrum(ref_path, cx, cy, region_size, percentage)
        if not ref_result:
            raise RuntimeError("Reference processing failed")
        ref_name, ref_waves, ref_intensities = ref_result
    
    ref_cube = load_fits(ref_path, mode="full")
    
    save_processed(
        ref_path,
        folder_path,
        "intensity",
        ref_cube,
        "_scaled",
        {'SCALE_FC': (1.0, 'Reference scaling factor')}
    )

    scaled_files = [ref_path]
    for file in files[1:]:
        try:
            # Load the full 3D cube
            full_cube = load_fits(file, mode="full")
            cx, cy = get_central_coords(file, shifts)
            if "_scaled.fits" in file:
                file_name = file
                target_waves = load_fits(file, mode="spectrum", x=cx, y=cy)
                target_intensities = np.arange(len(target_waves))
            else:
                target_result = process_spectrum(file, cx, cy, region_size, percentage)
                if not target_result:
                    logger.error(f"{file} processing failed")
                    continue
                else:
                    file_name, target_waves, target_intensities = target_result

            base_name = os.path.splitext(os.path.basename(file_name))[0] + '.fits'
            
            # Calculate or use existing scaling factor
            current_shifts = get_shifts(shifts, base_name)
            if current_shifts[5] != 1.0:  # If scaling exists
                factor = current_shifts[5]
            else:
                factor = calculate_scaling(ref_intensities, target_intensities)
                update_shifts(shifts, base_name, intensity_scaling=factor)
            
            # Apply scaling to full cube
            scaled_cube = full_cube * factor
            
            # Save scaled cube
            saved_path = save_processed(
                file,
                folder_path,
                "intensity",
                scaled_cube,
                "_scaled",
                {
                    'SCALE_FC': (factor, 'Intensity scaling factor'),
                    'REF_FILE': (os.path.basename(ref_path), 'Reference file')
                }
            )
            scaled_files.append(saved_path)
            
        except Exception as e:
            logger.error(f"Failed to process {file}: {str(e)}")
            continue

    save_shifts(shifts, os.path.join(folder_path, "shifts.csv"))

    # Plot final calibrated spectra
    final_spectra = []
    for file in files:
        try:
            # Load calibrated spectrum
            spectrum = load_fits(file)
            wavelengths = np.arange(len(spectrum))
            final_spectra.append((file, wavelengths, spectrum))
        except Exception as e:
            logger.error(f"Couldn't process {file} for plotting: {str(e)}")
    
    if final_spectra:
        plot_spectra(
            final_spectra,
            "Final Calibrated Spectra",
            os.path.join(folder_path, "plots", "final_calibrated_spectra.png")
        )

    return scaled_files

## Apply to H-alpha files
def apply_calibrations_to_ha_files(
    folder_path: str,
    shifts: Dict[str, Tuple],
    alignment_wavelength: int = 1,
    plot_spectra: bool = True
) -> List[str]:
    """Apply existing calibrations to matching H-alpha files.
    
    Transfers spatial shifts, wavelength shifts, and intensity scaling from Fe files
    to their corresponding H-alpha files.
    
    Args:
        folder_path (str): Path containing both Fe and H-alpha files
        shifts (Dict(str, Tuple)): Dictionary tracking calibration parameters
        alignment_wavelength (int): Wavelength index to use for alignment checks (default: 1)
        plot_spectra (bool): Whether to generate comparison plots (default: True)
    
    Raises:
        ValueError: If no Fe reference files found
    
    Returns:
        List[str]: List of paths to processed H-alpha files
    """

    logger.info("Applying calibrations to H-alpha files...")
    
    fe_files = [f for f in os.listdir(folder_path) if f.upper().endswith('_FE.FITS')]
    if not fe_files:
        raise ValueError("No Fe FITS files found to use as reference")

    # Prepare for plotting
    if plot_spectra:
        original_spectra = []
        aligned_spectra = []
        plot_dir = os.path.join(folder_path, "plots", "ha_calibration")
        os.makedirs(plot_dir, exist_ok=True)

    processed_files = []
    
    for fe_file in fe_files:
        fe_path = os.path.join(folder_path, fe_file)
        ha_path = find_matching_ha_file(fe_path)
        
        if not ha_path or not os.path.exists(ha_path):
            logger.warning(f"No matching H-alpha file found for {fe_file}")
            continue

        try:
            # Get original spectrum for plotting
            if plot_spectra:
                try:
                    original_data = load_fits(ha_path, mode="full")
                    if original_data.ndim == 3:
                        # For 3D data, take the specified wavelength slice
                        original_spec = original_data[alignment_wavelength]
                        # If the slice is still 2D, take mean or median
                        if original_spec.ndim == 2:
                            original_spec = np.median(original_spec, axis=0)
                        wavelengths = np.arange(len(original_spec))
                        original_spectra.append((ha_path, wavelengths, original_spec))
                    elif original_data.ndim == 2:
                        # For 2D data, take median row
                        original_spec = np.median(original_data, axis=0)
                        wavelengths = np.arange(len(original_spec))
                        original_spectra.append((ha_path, wavelengths, original_spec))
                    else:
                        logger.warning(f"Unexpected data dimensions {original_data.ndim} in {ha_path}")
                except Exception as e:
                    logger.error(f"Couldn't prepare original spectrum for {ha_path}: {str(e)}", exc_info=True)

            # Get shifts from corresponding Fe file
            base_name = os.path.splitext(os.path.basename(fe_path))[0] + '.fits'
            current_shifts = get_shifts(shifts, base_name)
            shift_y, shift_x, wavelength_shift, cx, cy, intensity_scaling = current_shifts

            # Load H-alpha cube
            ha_cube = load_fits(ha_path, mode="full")
            
            # Apply spatial shift
            aligned_cube = np.zeros_like(ha_cube)
            for i in range(ha_cube.shape[0]):
                aligned_cube[i] = transform_image(ha_cube[i], shift_x, shift_y)
            
            # Apply wavelength shift if needed
            if wavelength_shift != 0 and aligned_cube.ndim == 3:
                aligned_cube = np.roll(aligned_cube, wavelength_shift, axis=0)
                if wavelength_shift < 0:
                    aligned_cube[wavelength_shift:] = 0
                elif wavelength_shift > 0:
                    aligned_cube[:wavelength_shift] = 0
            
            # Apply intensity scaling
            scaled_cube = aligned_cube * intensity_scaling
            ensure_folder_structure(folder_path, True)
            
            # Save processed H-alpha file
            saved_path = save_processed(
                ha_path,
                folder_path,
                "h_alpha",
                scaled_cube,
                "_aligned",
                {
                    'SPATIAL': ('CALIBRATED', 'Spatial calibration applied'),
                    'SHIFT_X': (shift_x, 'X shift (pixels)'),
                    'SHIFT_Y': (shift_y, 'Y shift (pixels)'),
                    'WAVESHFT': (wavelength_shift, 'Wavelength shift'),
                    'SCALE_FC': (intensity_scaling, 'Intensity scaling factor'),
                    'REF_FILE': (fe_file, 'Reference Fe file')
                }
            )
            processed_files.append(saved_path)

        except Exception as e:
            logger.error(f"Failed to process {ha_path}: {str(e)}", exc_info=True)
            continue

    return processed_files