import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import threshold_otsu


def circleIdentify(filename: str):
    # Load file from documents
    #filepath1 = "RSM20241001T013504_0002_HA.fits"

    with fits.open(filename) as hdul:
        data = hdul[1].data
        # print(data.shape) # Print the shape of the compressed image data
        # Store the 70th row as a 2D array
        array = data[70, :, :]
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
    circles = houghTransform(edges, image_array)
    return circles


# How does the Hough transform work?

def houghTransform(edges: np.ndarray, array: np.ndarray):
    # Perform Circular Hough Transform for large circles
    hough_radii = np.arange(800, 1000, 1)  # Look for circle radii in the range 800-1000 (in the array, the radius of the sun is 925 for this file), with a step size of 3
    hough_res = hough_circle(edges, hough_radii)

    # Identify circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    # Visualize the detected circle
    fig, ax = plt.subplots()
    ax.imshow(array, cmap='gray')
    for center_y, center_x, radius in zip(cy, cx, radii):
        circ = plt.Circle((center_x, center_y), radius, color='red', fill=False)
        ax.add_patch(circ)
    plt.show()

    circles = np.array(list(zip(cx, cy, radii)))
    print(type(circles))
    return circles

# Transforming the second image to be lined up with the first

def transformDynArray(refArrayData: np.ndarray, dynArray: np.ndarray, dynArrayData: np.ndarray):
    # Calculate difference in array centroids
    transform = refArrayData[0] - dynArrayData[0]
    # Transform the image by the differece, to align images.
    transArray = sp.shift(dynArray, [transform[1], transform[0]], mode="nearest")
    return transArray

image1array = circleIdentify("RSM20241001T013504_0002_HA.fits")
image1data = preprocess(image1array)

image2array = circleIdentify("RSM20241001T013616_0003_HA.fits")
image2data = preprocess(image2array)

newarray = transformDynArray(image1data, image2array, image2data)
image3data = preprocess(newarray)
print(image3data)
