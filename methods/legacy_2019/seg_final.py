"""
seg_final.py -- standalone extraction of the FINAL fabric-segmentation algorithm
behind the reported mean IoU 0.88 on the 150-image set.

Provenance
----------
Source notebook : $PICKGLASS_NOTEBOOKS/2_Segmentation_Result_classical_CV.ipynb
Source cell     : cells[26] ("# Functions" > "## Image Segmentation related"), execution_count 269
0.88 result     : cells[39] (execution_count 275) ran
                  evaluate_algorithm(dataset[:150], annotations[:150], final_algorithm)
                  -> 'Mean IoU': 0.8843114933906611, 'Fallback Count': 14 (cells[40] raw summary).
                  No cell with execution_count in 270..275 redefines any of these functions.

Everything between the BEGIN/END VERBATIM markers is cells[26] source copied
byte-for-byte by build_seg_final.py (no algorithmic changes). Only the import
header above it and the segment()/warp/CLI code below it were added.

Entry point
-----------
    segment(image_path) -> (crop_bgr, box_points, used_fallback)
      crop_bgr      : fabric region perspective-warped to an upright rectangle,
                      cut from the ORIGINAL full-resolution image (added helper,
                      not part of the 2024 notebook code, which only returns a box)
      box_points    : (4,2) int array, box corners in original-image pixel coords
                      (exactly the `bounding_box` returned by final_algorithm)
      used_fallback : True when extract_contour returned status "fallback"
                      (centred 0.5W x 0.5H box)
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

# ======================= BEGIN VERBATIM: notebook cells[26] =======================
# Preprocessing function
def preprocess_image(image_path):
    """
    Preprocess the input image.
    - Resize to a fixed size
    - Convert to grayscale
    """
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    size = (1080, 1920) if h > w else (1920, 1080)
    resized_img = cv2.resize(img, size)
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    return img, resized_img, gray, (h, w, size)

# Shadow removal and thresholding
def apply_thresholding(gray_img, block_size=61, C=0):
    """
    Apply adaptive thresholding to remove shadows.
    """
    thresh = cv2.adaptiveThreshold(gray_img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    return thresh

# Filtering and convolution
def apply_custom_filter(thresh, k=30):
    """
    Apply a custom filter to the thresholded image to highlight fabric patterns.
    Steps:
    1. Convert the thresholded image values from [0,1] to [-1,1].
    2. Extract a (2k x 2k) patch around the image center as a filter kernel.
    3. Normalize and apply this kernel positively and negatively using filter2D.
    4. Dilate and combine the results to enhance fabric pattern detection.
    
    Parameters:
        thresh (numpy.ndarray): Thresholded image with values {0,1}.
        k (int): Half-size of the central patch.
        
    Returns:
        numpy.ndarray: An image with enhanced fabric patterns.
    """
    # Ensure floating-point arithmetic for correct [-1, 1] scaling
    thresh_float = thresh.astype(np.float32)
    thresh_scaled = thresh_float * 2.0 - 1.0  # range: [-1, 1]

    # Get image dimensions
    height, width = thresh_scaled.shape
    
    # Extract the central (2k x 2k) patch as the filter kernel
    center_y, center_x = height // 2, width // 2
    filti = thresh_scaled[center_y - k:center_y + k, center_x - k:center_x + k]

    # Normalize the kernel for value scaling (as in the original code)
    filti = filti * 255.0 / (4.0 * k * k)  # This maintains the original normalization

    # Apply the positive and negative filters to the image
    filtered_pos = cv2.filter2D(thresh_scaled, cv2.CV_8UC3, filti / 2)
    filtered_neg = cv2.filter2D(thresh_scaled, cv2.CV_8UC3, -filti / 2)

    # Dilate both filtered outputs to strengthen detected patterns
    kernel = np.ones((30, 30), np.uint8)
    filtered_pos = cv2.dilate(filtered_pos, kernel, iterations=1)
    filtered_neg = cv2.dilate(filtered_neg, kernel, iterations=1)

    # Combine the positive and negative filtered results
    filtered_combined = filtered_pos + filtered_neg

    return filtered_combined

def apply_bilateral_filtering(thresh):
    """
    Apply bilateral filtering and further process the binary image.
    
    Args:
    - thresh (numpy.ndarray): Binary thresholded image.
    
    Returns:
    - vis (numpy.ndarray): Processed image after bilateral filtering.
    """
    # Convert binary threshold to 8-bit image for processing
    vis = thresh * 255
    
    # Apply bilateral filtering to reduce noise while preserving edges
    vis = cv2.bilateralFilter(vis, 3, 5, 5)
    
    # Perform dilation to enhance features
    kernel = np.ones((7, 7), np.uint8)
    vis = cv2.dilate(vis, kernel, iterations=1)
    
    # Apply multiple blurring operations for smoothing
    for _ in range(6):
        vis = cv2.blur(vis, (25, 25))
    
    return vis

def combine_custom_and_bilateral(filtered_img3, bilateral_img):
    """
    Combine the custom filtered image with the bilateral filtered image.
    
    Args:
    - filtered_img3 (numpy.ndarray): Output from the custom filter.
    - bilateral_img (numpy.ndarray): Output from the bilateral filtering.
    
    Returns:
    - combined_img (numpy.ndarray): Combined image ready for K-means clustering.
    """
    # Smooth the custom filter output
    filtered_img3 = cv2.blur(filtered_img3, (30, 30))
    
    # Combine the two images
    combined_img = (filtered_img3 + bilateral_img) / 2
    
    return combined_img

def post_process_for_kmeans(thresh, filtered_img3):
    """
    Refactored post-processing pipeline:
    1. Apply bilateral filtering.
    2. Combine custom filtered and bilateral filtered outputs.
    """
    # Apply bilateral filtering
    bilateral_img = apply_bilateral_filtering(thresh)
    
    # Combine with custom filtering output
    combined_img = combine_custom_and_bilateral(filtered_img3, bilateral_img)
    
    return combined_img
    
def apply_kmeans(filtered_final, K=2):
    """
    Perform K-means clustering to segment the image into clusters.

    Args:
    - filtered_final (numpy.ndarray): Processed image ready for clustering.
    - K (int): Number of clusters (default is 2 for binary segmentation).

    Returns:
    - segmented_img (numpy.ndarray): Binary segmented image with pixel values 0 or 255.
    """
    # Step 1: Reshape the image into a 1D array for clustering
    pixel_values = filtered_final.reshape((-1, 1)).astype(np.float32)
    
    # Step 2: Define criteria for K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Step 3: Perform K-means clustering
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Step 4: Create binary result based on cluster centers
    binary_result = np.zeros_like(labels, dtype=np.uint8)
    if centers[0] < centers[1]:
        binary_result[labels == 0] = 0
        binary_result[labels == 1] = 255
    else:
        binary_result[labels == 0] = 255
        binary_result[labels == 1] = 0
    
    # Step 5: Reshape the result back to the original image shape
    segmented_img = binary_result.reshape(filtered_final.shape).astype(np.uint8)
    
    return segmented_img

def extract_contour(segmented_img, resized_img, original_size, fallback_area_factor=1/7, return_null_box_if_no_contour=False):
    """
    Extract contours from the segmented image to compute the bounding box.
    If the bounding box area is too small or no valid contours are found, a fallback box is used.

    Args:
    - segmented_img (numpy.ndarray): Binary image from K-means segmentation.
    - resized_img (numpy.ndarray): Resized version of the original image.
    - original_size (tuple): Tuple containing original (height, width, resized dimensions).
    - fallback_area_factor (float): Minimum area of the bounding box as a fraction 
      of the total image area. If the box area is below this value, a fallback box 
      (0.5x the image width and height, centered) is used. Default is 1/6.
    - return_null_box_if_no_contour (bool): If True, return an empty box ([0, 0], [0, 0], [0, 0], [0, 0]) 
      when no contours are found instead of a fallback box.

    
    Returns:
    - original_img (numpy.ndarray): Image with the bounding box drawn.
    - box (numpy.ndarray): Array of points defining the bounding box.
    """
    def generate_fallback_box(original_size, resized_img):
        """Generate a fallback box and draw it on the resized image."""
        original_h, original_w = original_size[:2]
        center_x, center_y = original_w / 2, original_h / 2
        fallback_width, fallback_height = 0.5 * original_w, 0.5 * original_h
        fallback_box = np.array([
            [center_x - fallback_width / 2, center_y - fallback_height / 2],
            [center_x + fallback_width / 2, center_y - fallback_height / 2],
            [center_x + fallback_width / 2, center_y + fallback_height / 2],
            [center_x - fallback_width / 2, center_y + fallback_height / 2],
        ], dtype=int)

        # Resize the image to original dimensions and draw the fallback box
        original_h, original_w, resized_size = original_size
        original_img = cv2.resize(resized_img, (original_w, original_h))
        cv2.drawContours(original_img, [fallback_box], -1, (0, 255, 0), 3)

        return original_img, fallback_box

    def generate_empty_box():
        """Generate an empty box."""
        return np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=int)
        
    def find_largest_suitable_contour(contours, image_shape):
        """Find the largest contour close to the center with reasonable area."""
        a, b = image_shape
        largest_contour = None
        max_area = 0
        min_dist_to_center = float("inf")
        
        for c in contours:
            area = cv2.contourArea(c)
            if 100 < len(c) < 1500 and area < 0.5 * a * b:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    dist_to_center = math.sqrt((cx - b / 2) ** 2 + (cy - a / 2) ** 2)
                    if area > max_area and dist_to_center < min_dist_to_center:
                        max_area = area
                        min_dist_to_center = dist_to_center
                        largest_contour = c
        return largest_contour

    def scale_box_to_original(box, original_size, resized_size):
        """Scale the bounding box back to the original image dimensions."""
        original_h, original_w = original_size[:2]
        resized_w, resized_h = resized_size
        scale_x = original_w / resized_w
        scale_y = original_h / resized_h
        return np.array([[int(x * scale_x), int(y * scale_y)] for x, y in box])

    # Step 1: Find contours in the segmented image
    contours, _ = cv2.findContours(segmented_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fallback if no contours found
    if not contours:
        if return_null_box_if_no_contour:
            print("No contours found. Returning empty box.")
            return resized_img, generate_empty_box(), "empty"
        print("No contours found. Using fallback box.")
        fallback_img, fallback_box = generate_fallback_box(original_size, resized_img)
        return fallback_img, fallback_box, "fallback"

    # Step 2: Find the largest suitable contour
    largest_contour = find_largest_suitable_contour(contours, segmented_img.shape)
    
    # Fallback if no suitable contour found
    if largest_contour is None:
        if return_null_box_if_no_contour:
            print("No suitable contour found. Returning empty box.")
            return resized_img, generate_empty_box(), "empty"
        print("No suitable contour found. Using fallback box.")
        fallback_img, fallback_box = generate_fallback_box(original_size, resized_img)
        return fallback_img, fallback_box, "fallback"

    # Step 3: Process the contour and extract the bounding box
    black = np.zeros_like(segmented_img)
    cv2.drawContours(black, [largest_contour], -1, [255], 1)
    kernel = np.ones((100, 100), np.uint8)
    dilated_img = cv2.dilate(black, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    smallest_contour = min(contours, key=cv2.contourArea, default=None)

    # Fallback if smallest contour is not valid
    if smallest_contour is None:
        if return_null_box_if_no_contour:
            print("No smallest contour found. Returning empty box.")
            return resized_img, generate_empty_box(), "empty"
        print("No smallest contour found. Using fallback box.")
        fallback_img, fallback_box = generate_fallback_box(original_size, resized_img)
        return fallback_img, fallback_box, "fallback"

    filled_img = np.zeros_like(black)
    cv2.drawContours(filled_img, [smallest_contour], -1, [255], -1)
    final_dilated_img = cv2.dilate(filled_img, kernel, iterations=1)

    contours, _ = cv2.findContours(final_dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect).astype(int)

    # Step 4: Scale the box to the original image size
    box = scale_box_to_original(box, original_size, original_size[2])

    # Step 5: Check if the box area is too small, and use fallback if needed
    box_area = cv2.contourArea(box)
    image_area = original_size[0] * original_size[1]
    if box_area < image_area * fallback_area_factor:
        print("Fallback box used due to small area.")
        fallback_img, fallback_box = generate_fallback_box(original_size, resized_img)
        return fallback_img, fallback_box, "fallback"

    # Step 6: Resize the image back to the original dimensions and draw the box
    original_h, original_w = original_size[:2]
    original_img = cv2.resize(resized_img, (original_w, original_h))
    cv2.drawContours(original_img, [box], -1, (0, 255, 0), 3)

    return original_img, box, "valid"


def final_algorithm(image_path, fallback_area_factor=1/7, return_null_box_if_no_contour=False):
    """
    Execute the complete pipeline for extracting the bounding box:
    1. Preprocess the image.
    2. Apply adaptive thresholding.
    3. Apply a custom filter to enhance fabric patterns.
    4. Post-process the image for k-means clustering.
    5. Perform k-means clustering for segmentation.
    6. Extract the contour and bounding box from the segmented image.

    Args:
    - image_path (str): Path to the input image.
    - fallback_area_factor (float): Minimum area of the bounding box as a fraction 
      of the total image area. If the box area is below this value, a fallback box 
      (0.5x the image width and height, centered) is used. Default is 1/6.

    Returns:
    - final_img (numpy.ndarray): Image with the bounding box drawn.
    - bounding_box (numpy.ndarray): Array of points defining the bounding box.
    """
    np.random.seed(42)
    
    # Step 1: Preprocess the image (resize and convert to grayscale)
    img, resized_img, gray, size_info = preprocess_image(image_path)
    
    # Step 2: Apply adaptive thresholding to remove shadows
    thresh = apply_thresholding(gray)
    
    # Step 3: Apply a custom filter to enhance periodic patterns
    custom_filtered_img = apply_custom_filter(thresh)
    
    # Step 4: Post-process the filtered image for clustering
    filtered_final = post_process_for_kmeans(thresh, custom_filtered_img)
    
    # Step 5: Perform k-means clustering to segment the image
    segmented_img = apply_kmeans(filtered_final)
    
    # Step 6: Extract contours and compute the bounding box, passing fallback_area_factor
    final_img, bounding_box, status = extract_contour(segmented_img, resized_img, size_info, fallback_area_factor, return_null_box_if_no_contour)
    
    return final_img, bounding_box, status
# ======================== END VERBATIM: notebook cells[26] ========================


# ---------------------------------------------------------------------------
# Added for standalone use (NOT part of the notebook code).
# ---------------------------------------------------------------------------
def _order_quad(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def warp_box(image_bgr, box_points):
    """Perspective-warp the quadrilateral `box_points` of `image_bgr` to an upright
    rectangle (the report's 'perspective-warp the fabric region' step; no 2019/2024
    implementation of this step was found, so this is a minimal helper)."""
    quad = _order_quad(box_points)
    tl, tr, br, bl = quad
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if w < 2 or h < 2:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image_bgr, M, (w, h))


def segment(image_path, fallback_area_factor=1/7, cv_seed=None):
    """Run the final 2024 algorithm (defaults identical to the 0.88 run) and crop.

    cv_seed: optional cv2.setRNGSeed value. The notebook only called
    np.random.seed(42), which does NOT seed cv2.kmeans(KMEANS_RANDOM_CENTERS);
    default None keeps the notebook behaviour.
    Returns (crop_bgr, box_points, used_fallback).
    """
    if cv_seed is not None:
        cv2.setRNGSeed(int(cv_seed))
    _final_img, box, status = final_algorithm(
        image_path, fallback_area_factor=fallback_area_factor,
        return_null_box_if_no_contour=False)
    original = cv2.imread(image_path)
    crop = warp_box(original, box)
    return crop, np.asarray(box), status == "fallback"


def _visualise(image_path, box, used_fallback, crop, out_path, gt_box=None):
    """Side-by-side: original with predicted box (green; red if fallback) [+ GT in blue], and crop."""
    img = cv2.imread(image_path)
    vis = img.copy()
    t = max(3, int(round(max(img.shape[:2]) / 400)))
    if gt_box is not None:
        cv2.polylines(vis, [np.asarray(gt_box, dtype=np.int32).reshape(-1, 1, 2)], True, (255, 0, 0), t)
    color = (0, 0, 255) if used_fallback else (0, 255, 0)
    cv2.polylines(vis, [np.asarray(box, dtype=np.int32).reshape(-1, 1, 2)], True, color, t)
    H = 900
    left = cv2.resize(vis, (int(vis.shape[1] * H / vis.shape[0]), H))
    if crop is not None and crop.size:
        right = cv2.resize(crop, (max(1, int(crop.shape[1] * H / crop.shape[0])), H))
    else:
        right = np.zeros((H, H // 2, 3), np.uint8)
    canvas = np.concatenate([left, np.full((H, 10, 3), 255, np.uint8), right], axis=1)
    cv2.imwrite(out_path, canvas)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Final 2024 fabric segmentation (notebook cells[26]).")
    ap.add_argument("images", nargs="+", help="image paths")
    ap.add_argument("--out", default=None, help="directory for <stem>_crop.png and <stem>_vis.jpg")
    ap.add_argument("--fallback-area-factor", type=float, default=1/7)
    ap.add_argument("--cv-seed", type=int, default=None)
    args = ap.parse_args(argv)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    for p in args.images:
        crop, box, fb = segment(p, args.fallback_area_factor, args.cv_seed)
        print(f"{p}\tfallback={fb}\tbox={box.tolist()}\tcrop={None if crop is None else crop.shape}")
        if args.out:
            stem = os.path.splitext(os.path.basename(p))[0].strip().replace(" ", "_")
            if crop is not None:
                cv2.imwrite(os.path.join(args.out, stem + "_crop.png"), crop)
            _visualise(p, box, fb, crop, os.path.join(args.out, stem + "_vis.jpg"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
