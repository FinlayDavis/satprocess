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
from typing import Dict, Tuple, Optional, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Shift Management Functions
def clean_filename(filename: str) -> str:
    """Strip alignment suffixes to get primary filename"""
    suffixes = ['_aligned', '_wave', '_scaled', '_spatial']
    for suffix in suffixes:
        filename = filename.replace(suffix, '')
    return os.path.splitext(filename)[0] + '.fits'  # Ensure .fits extension

def load_shifts(shifts_file: str) -> Dict[str, Tuple]:
    """Load shifts from CSV file, keeping only primary filenames"""
    shifts = {}
    try:
        if os.path.exists(shifts_file):
            with open(shifts_file, mode="r") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if not row:
                        continue
                    filename = clean_filename(row[0])
                    try:
                        # Ensure we have exactly 6 values (padding with defaults if needed)
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
    """Save current shifts to file, maintaining only primary filenames"""
    try:
        os.makedirs(os.path.dirname(shifts_file), exist_ok=True)
        with open(shifts_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "filename", "shift_y", "shift_x", "wavelength_shift", 
                "cx", "cy", "intensity_scaling"
            ])
            
            # Sort by filename for consistent ordering
            for filename in sorted(shifts.keys()):
                values = shifts[filename]
                writer.writerow([filename] + list(values))
    except Exception as e:
        logger.error(f"Error saving shifts: {str(e)}")

def get_shifts(shifts: Dict[str, Tuple], filename: str) -> Tuple:
    """Get shifts for a file (using cleaned filename) with defaults"""
    clean_name = clean_filename(filename)
    return shifts.get(clean_name, (0.0, 0.0, 0, 0.0, 0.0, 1.0))

def update_shifts(shifts: Dict[str, Tuple], filename: str, **updates):
    """Update specific shift values for a file (using cleaned filename)"""
    clean_name = clean_filename(filename)
    current = list(get_shifts(shifts, clean_name))
    fields = ['shift_y', 'shift_x', 'wavelength_shift', 'cx', 'cy', 'intensity_scaling']
    
    for field, value in updates.items():
        if field in fields:
            idx = fields.index(field)
            current[idx] = value
    
    shifts[clean_name] = tuple(current)

# FITS Processing Functions
def ensure_folder_structure(base_path: str):
    """Create required folder structure"""
    folders = ["Spatial", "Wavelength", "Intensity"]
    for folder in folders:
        os.makedirs(os.path.join(base_path, "AlignedImages", folder), exist_ok=True)

def load_fits(file_path: str, mode: str = "full", **kwargs) -> np.ndarray:
    """Load data from FITS file with different modes"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    with fits.open(file_path, memmap=True) as hdul:
        if len(hdul) < 2:
            raise ValueError("FITS file missing data extension")
        
        data = hdul[1].data
        
        # Handle 1D spectra (from wavelength calibration)
        if data.ndim == 1:
            if mode != "full":
                raise ValueError("1D data only supports 'full' mode")
            return data
            
        # Handle 3D cubes
        elif data.ndim == 3:
            if mode == "full":
                return data
            elif mode == "slice":
                wavelength = kwargs.get('wavelength')
                if wavelength is None or not 0 <= wavelength < data.shape[0]:
                    raise ValueError("Invalid wavelength index")
                return data[wavelength]
            elif mode == "spectrum":
                x, y = kwargs.get('x_coord'), kwargs.get('y_coord')
                if None in (x, y) or not (0 <= x < data.shape[1] and 0 <= y < data.shape[2]):
                    raise ValueError("Invalid coordinates")
                return data[:, x, y]
            elif mode == "region":
                cx, cy, size = kwargs.get('center_x'), kwargs.get('center_y'), kwargs.get('size', 100)
                half = size // 2
                x_start, x_end = max(0, cx - half), min(data.shape[2], cx + half)
                y_start, y_end = max(0, cy - half), min(data.shape[1], cy + half)
                return data[:, y_start:y_end, x_start:x_end]
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            raise ValueError(f"Unsupported data dimensionality: {data.ndim}")

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
        original_path: Path to original file
        base_path: Base directory path
        output_type: One of 'spatial', 'wavelength', 'intensity'
        data: Processed data to save
        suffix: Filename suffix
        metadata: Dictionary of header updates
    
    Returns:
        Path to saved file
    """
    output_folder = os.path.join(base_path, "AlignedImages", output_type.capitalize())
    os.makedirs(output_folder, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}{suffix}.fits")

    with fits.open(original_path) as hdul:
        # Update headers
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, tuple):
                    hdul[0].header[key] = value[0], value[1]  # (value, comment)
                else:
                    hdul[0].header[key] = value
        
        # Update data
        hdul[1].data = data
        hdul.writeto(output_path, overwrite=True)
    
    logger.info(f"Saved {output_type} data to {output_path}")
    return output_path

def find_matching_ha_file(fe_file: str) -> Optional[str]:
    """Find matching H-alpha file for a given Fe file"""
    base_name = os.path.basename(fe_file)
    if '_FE.fits' in base_name:
        ha_file = base_name.replace('_FE.fits', '_HA.fits')
        return os.path.join(os.path.dirname(fe_file), ha_file)
    return None

# Image Processing Functions
def preprocess_image(image: np.ndarray, sigma: float = 2) -> np.ndarray:
    """Prepare image for circle detection"""
    blurred = sp.gaussian_filter(image, sigma=sigma)
    thresh = threshold_otsu(blurred)
    binary = blurred > thresh
    return canny(binary, sigma=1)

def detect_circles(edges: np.ndarray, min_radius: int = 800, max_radius: int = 1000) -> np.ndarray:
    """Detect circles using Hough transform"""
    radii = np.arange(min_radius, max_radius, 1)
    hough_res = hough_circle(edges, radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, radii, total_num_peaks=1)
    return np.array(list(zip(cx, cy, radii)))

def transform_image(image: np.ndarray, dx: float, dy: float, mode: str = "nearest") -> np.ndarray:
    """Shift image by specified amount"""
    return sp.shift(image, shift=[dy, dx], mode=mode)

def extract_median_pixels(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    size: int = 100,
    percentage: float = 1
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Extract median percentage of pixels from square region"""
    half = size // 2
    y_start, y_end = max(0, center_y - half), min(image.shape[0], center_y + half)
    x_start, x_end = max(0, center_x - half), min(image.shape[1], center_x + half)
    
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
    """Calculate spectral shift using cross-correlation"""
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
    """Align target spectrum to reference"""
    shift = calculate_shift(ref_intensities, target_intensities)
    shifted = np.roll(target_intensities, shift)
    
    if shift < 0:
        shifted[shift:] = 0
    elif shift > 0:
        shifted[:shift] = 0
    
    interp_func = interp1d(target_wavelengths, shifted, kind='linear', fill_value='extrapolate')
    return interp_func(ref_wavelengths)

def calculate_scaling(ref_spec: np.ndarray, target_spec: np.ndarray) -> float:
    """Calculate intensity scaling factor"""
    mask = (ref_spec != 0) & (target_spec != 0)
    if not np.any(mask):
        return 1.0
    
    ref_integral = np.trapezoid(ref_spec[mask])
    target_integral = np.trapezoid(target_spec[mask])
    return ref_integral / target_integral if target_integral != 0 else 1.0

# Plotting Functions
def plot_spectra(
    spectra_data: List[Tuple[str, np.ndarray, np.ndarray]], 
    title: str,
    save_path: str,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot multiple spectra on the same figure
    
    Args:
        spectra_data: List of (filename, wavelengths, intensities) tuples
        title: Plot title
        save_path: Where to save the plot
        figsize: Figure dimensions
    """
    plt.figure(figsize=figsize)
    for file_path, wavelengths, intensities in spectra_data:
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

# Pipeline Functions
def process_spectrum(
    file_path: str,
    shifts: Dict[str, Tuple],
    region_size: int,
    percentage: float
) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """Helper function to process spectrum from file"""
    try:
        # For wavelength-aligned files, we can just load the 1D spectrum directly
        if "_wave.fits" in file_path:
            spectrum = load_fits(file_path)
            wavelengths = np.arange(len(spectrum))
            return file_path, wavelengths, spectrum
            
        # Original 3D cube processing for spatial calibration
        base_name = os.path.splitext(os.path.basename(file_path))[0] + '.fits'
        current_shifts = get_shifts(shifts, base_name)
        
        # Get center coordinates
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
        
        # Load 3D region
        region_3d = load_fits(
            file_path,
            mode="region",
            center_x=int(cx),
            center_y=int(cy),
            size=region_size
        )
        
        # Get median pixels
        alignment_slice = load_fits(file_path, mode="slice", wavelength=1)
        median_x, median_y, start_x, start_y = extract_median_pixels(
            alignment_slice, int(cx), int(cy), region_size, percentage
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
    min_radius: int = 800,
    max_radius: int = 1000
) -> List[str]:
    """Perform spatial alignment of all images"""
    logger.info("Starting spatial calibration...")
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.fits')]
    if not files:
        raise ValueError("No FITS files found")
    if reference_index >= len(files):
        raise ValueError("Invalid reference index")

    # Process reference image
    ref_path = os.path.join(folder_path, files[reference_index])
    ref_slice = load_fits(ref_path, mode="slice", wavelength=alignment_wavelength)
    ref_edges = preprocess_image(ref_slice)
    ref_circles = detect_circles(ref_edges, min_radius, max_radius)
    ref_center = ref_circles[0][:2]
    logger.info(f"Reference center: {ref_center}")

    aligned_files = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            base_name = os.path.splitext(file)[0] + '.fits'
            current_shifts = get_shifts(shifts, base_name)
            
            # Load alignment slice
            alignment_slice = load_fits(file_path, mode="slice", wavelength=alignment_wavelength)
            
            # Calculate or use existing shifts
            if current_shifts[3:5] != (0.0, 0.0):  # If center exists
                cx, cy = current_shifts[3:5]
                shift_x = ref_center[0] - cx
                shift_y = ref_center[1] - cy
            else:
                edges = preprocess_image(alignment_slice)
                circles = detect_circles(edges, min_radius, max_radius)
                cx, cy, _ = circles[0]
                shift_x = ref_center[0] - cx
                shift_y = ref_center[1] - cy
                update_shifts(shifts, base_name, cx=cx, cy=cy, shift_x=shift_x, shift_y=shift_y)
            
            # Apply shift to full cube
            full_cube = load_fits(file_path, mode="full")
            aligned_cube = np.zeros_like(full_cube)
            for i in range(full_cube.shape[0]):
                aligned_cube[i] = transform_image(full_cube[i], shift_x, shift_y)
            
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
                    'SHIFT_Y': (shift_y, 'Y shift (pixels)')
                }
            )
            aligned_files.append(saved_path)
            
        except Exception as e:
            logger.error(f"Failed to process {file}: {str(e)}")
            continue

    save_shifts(shifts, os.path.join(folder_path, "shifts.csv"))

    # Plot original spectra
    original_spectra = []
    for file in files:
        try:
            # Get spectrum from center region
            result = process_spectrum(
                os.path.join(folder_path, file),
                shifts,
                region_size=100,
                percentage=1
            )
            if result:
                original_spectra.append(result)
        except Exception as e:
            logger.error(f"Couldn't process {file} for plotting: {str(e)}")
    
    if original_spectra:
        plot_spectra(
            original_spectra,
            "Original Spectra (Spatially Aligned)",
            os.path.join(folder_path, "plots", "spatial_alignment_spectra.png")
        )

    return aligned_files

def wavelength_calibration(
    folder_path: str,
    shifts: Dict[str, Tuple],
    region_size: int = 100,
    percentage: float = 1,
    max_workers: int = 4
) -> List[str]:
    """Align spectra in wavelength space"""
    logger.info("Starting wavelength calibration...")
    
    spatial_folder = os.path.join(folder_path, "AlignedImages", "Spatial")
    files = [os.path.join(spatial_folder, f) for f in os.listdir(spatial_folder) if f.endswith('_aligned.fits')]
    if not files:
        raise ValueError("No spatially aligned files found")

    # Process reference
    ref_path = files[0]
    ref_result = process_spectrum(ref_path, shifts, region_size, percentage)
    if not ref_result:
        raise RuntimeError("Reference processing failed")
    
    ref_name, ref_waves, ref_intensities = ref_result
    save_processed(
        ref_path,
        folder_path,
        "wavelength",
        ref_intensities,
        "_wave",
        {'WAVE_REF': ('REFERENCE', 'Reference spectrum')}
    )

    aligned_files = [ref_path]
    for file in files[1:]:
        try:
            result = process_spectrum(file, shifts, region_size, percentage)
            if not result:
                continue
                
            file_name, target_waves, target_intensities = result
            base_name = os.path.splitext(os.path.basename(file_name))[0] + '.fits'
            
            # Calculate or use existing wavelength shift
            current_shifts = get_shifts(shifts, base_name)
            if current_shifts[2] != 0:  # If wavelength shift exists
                shift = current_shifts[2]
            else:
                shift = calculate_shift(ref_intensities, target_intensities)
                update_shifts(shifts, base_name, wavelength_shift=shift)
            
            # Align spectrum
            aligned = align_spectrum(
                ref_waves, ref_intensities, target_waves, target_intensities
            )
            
            # Save aligned spectrum
            saved_path = save_processed(
                file,
                folder_path,
                "wavelength",
                aligned,
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

    # Plot wavelength-aligned spectra
    wave_aligned_spectra = []
    for file in aligned_files:  # Use the aligned files, not the original files
        try:
            # Load the wavelength-aligned spectrum
            spectrum = load_fits(file)
            
            # Ensure we're working with 1D data
            if spectrum.ndim > 1:
                # If it's still 3D, take the first spectrum (or handle appropriately for your case)
                spectrum = spectrum[0] if spectrum.ndim == 3 else spectrum
                if spectrum.ndim > 1:
                    # If still 2D, take the first row or column
                    spectrum = spectrum[0] if spectrum.shape[0] < spectrum.shape[1] else spectrum[:, 0]
            
            wavelengths = np.arange(len(spectrum))
            wave_aligned_spectra.append((file, wavelengths, spectrum))
        except Exception as e:
            logger.error(f"Couldn't process {file} for plotting: {str(e)}")
    
    if wave_aligned_spectra:
        plot_spectra(
            wave_aligned_spectra,
            "Spectra After Wavelength Calibration",
            os.path.join(folder_path, "plots", "wavelength_aligned_spectra.png")
        )
    return aligned_files

def intensity_calibration(
    folder_path: str,
    shifts: Dict[str, Tuple],
    region_size: int = 100,
    percentage: float = 1,
    max_workers: int = 4
) -> List[str]:
    """Normalize intensities across all spectra"""
    logger.info("Starting intensity calibration...")
    
    wave_folder = os.path.join(folder_path, "AlignedImages", "Wavelength")
    files = [os.path.join(wave_folder, f) for f in os.listdir(wave_folder) if f.endswith('_wave.fits')]
    if not files:
        raise ValueError("No wavelength-aligned files found")

    # Process reference
    ref_path = files[0]
    ref_result = process_spectrum(ref_path, shifts, region_size, percentage)
    if not ref_result:
        raise RuntimeError("Reference processing failed")
    
    ref_name, ref_waves, ref_intensities = ref_result
    save_processed(
        ref_path,
        folder_path,
        "intensity",
        ref_intensities,
        "_scaled",
        {'SCALE_FC': (1.0, 'Reference scaling factor')}
    )

    scaled_files = [ref_path]
    for file in files[1:]:
        try:
            result = process_spectrum(file, shifts, region_size, percentage)
            if not result:
                continue
                
            file_name, target_waves, target_intensities = result
            base_name = os.path.splitext(os.path.basename(file_name))[0] + '.fits'
            
            # Calculate or use existing scaling factor
            current_shifts = get_shifts(shifts, base_name)
            if current_shifts[5] != 1.0:  # If scaling exists
                factor = current_shifts[5]
            else:
                factor = calculate_scaling(ref_intensities, target_intensities)
                update_shifts(shifts, base_name, intensity_scaling=factor)
            
            # Apply scaling
            scaled = target_intensities * factor
            
            # Save scaled spectrum
            saved_path = save_processed(
                file,
                folder_path,
                "intensity",
                scaled,
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

def apply_calibrations_to_ha_files(
    folder_path: str,
    shifts: Dict[str, Tuple],
    alignment_wavelength: int = 1
) -> List[str]:
    """Apply existing calibrations to matching H-alpha files"""
    logger.info("Applying calibrations to H-alpha files...")
    
    files = [f for f in os.listdir(folder_path) if f.endswith('_FE.fits')]
    if not files:
        raise ValueError("No Fe FITS files found")

    processed_files = []
    for fe_file in files:
        fe_path = os.path.join(folder_path, fe_file)
        ha_path = find_matching_ha_file(fe_path)
        
        if not ha_path or not os.path.exists(ha_path):
            logger.warning(f"No matching H-alpha file found for {fe_file}")
            continue

        try:
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
            
            # Apply wavelength shift if needed (for 3D cubes)
            if wavelength_shift != 0 and aligned_cube.ndim == 3:
                aligned_cube = np.roll(aligned_cube, wavelength_shift, axis=0)
                if wavelength_shift < 0:
                    aligned_cube[wavelength_shift:] = 0
                elif wavelength_shift > 0:
                    aligned_cube[:wavelength_shift] = 0
            
            # Apply intensity scaling
            scaled_cube = aligned_cube * intensity_scaling
            
            # Save processed H-alpha file
            saved_path = save_processed(
                ha_path,
                folder_path,
                "spatial",
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
            logger.error(f"Failed to process {ha_path}: {str(e)}")
            continue

    return processed_files

if __name__ == "__main__":
    try:
        folder_path = "TestImages"
        shifts_file = os.path.join(folder_path, "shifts.csv")
        
        # Ensure folder structure exists
        ensure_folder_structure(folder_path)
        os.makedirs(os.path.join(folder_path, "plots"), exist_ok=True)
        
        # Load or initialize shifts
        shifts = load_shifts(shifts_file)
        
        # Run full calibration pipeline on Fe files
        spatial_results = spatial_calibration(
            folder_path,
            shifts,
            alignment_wavelength=40,
            reference_index=0,
            min_radius=900,
            max_radius=950
        )
        
        wavelength_results = wavelength_calibration(
            folder_path,
            shifts,
            region_size=100,
            percentage=1
        )
        
        intensity_results = intensity_calibration(
            folder_path,
            shifts,
            region_size=100,
            percentage=1
        )
        
        # Apply calibrations to matching H-alpha files
        ha_results = apply_calibrations_to_ha_files(folder_path, shifts)
        
        logger.info("Calibration completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)