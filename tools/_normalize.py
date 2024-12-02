import open3d as o3d
import os
import numpy as np


def scale_to_bounding_box(points):
    """
    Scales a point cloud to fit within the bounding box [0, 1], and returns normalization parameters.

    Args:
        points (np.array): A 3D point cloud with shape (N, 3), where N is the number of points.

    Returns:
        tuple: (scaled_points, normalization_params)
            - scaled_points: Scaled point cloud with points scaled to fit within [0, 1].
            - normalization_params: Dictionary containing the centroid, scale factor, and min/max used for normalization.
    """

    # Step 1: Translate points to the origin (center the point cloud)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    # Step 2: Find the bounding box (min and max along each axis)
    min_coords = np.min(points_centered, axis=0)
    max_coords = np.max(points_centered, axis=0)

    # Step 3: Find the size of the bounding box in each dimension (width, height, depth)
    scale_factor = np.max(max_coords - min_coords)  # The largest dimension determines the scale

    # Step 4: Scale points to fit within [0, 1] by dividing by the bounding box size
    points_scaled = points_centered / scale_factor

    # Step 5: Translate the points so that they fit within [0, 1]
    # Find the new min and max (after scaling)
    min_scaled = np.min(points_scaled, axis=0)
    max_scaled = np.max(points_scaled, axis=0)

    # Shift the points to ensure they lie within the unit cube [0, 1]
    points_scaled = points_scaled - min_scaled  # Translate to [0, scale_factor]

    # Normalization parameters: centroid, scale factor, min/max values for translation
    normalization_params = {
        "centroid": centroid,
        "scale_factor": scale_factor,
        "min_scaled": min_scaled,
        "max_scaled": max_scaled
    }

    return points_scaled, normalization_params


def apply_normalization(partial_points, normalization_params):
    """
    Applies the stored normalization parameters to a partial point cloud to ensure consistent scaling.

    Args:
        partial_points (np.array): A partial 3D point cloud with shape (N, 3).
        normalization_params (dict): Dictionary containing the centroid, scale factor, and min/max used for normalization.

    Returns:
        np.array: Normalized partial point cloud with points scaled to fit within [0, 1].
    """
    # Extract normalization parameters
    centroid = normalization_params["centroid"]
    scale_factor = normalization_params["scale_factor"]
    min_scaled = normalization_params["min_scaled"]

    # Step 1: Translate the partial point cloud to the origin (center it using the original centroid)
    points_centered = partial_points - centroid

    # Step 2: Scale the points using the same scale factor
    points_scaled = points_centered / scale_factor

    # Step 3: Translate the scaled points to ensure they fit within [0, 1]
    points_scaled = points_scaled - min_scaled  # Translate to [0, scale_factor]

    return points_scaled


# Example usage:

# Original point cloud
points = np.random.rand(1000, 3) * 10  # Example: 3D point cloud (randomly generated)

# Scale and get normalization parameters
scaled_points, normalization_params = scale_to_bounding_box(points)

print("Scaled Points:\n", scaled_points)
print("Normalization Params:\n", normalization_params)

# Apply the same normalization to a partial point cloud (same object)
partial_points = np.random.rand(500, 3) * 10  # Example: partial point cloud
normalized_partial_points = apply_normalization(partial_points, normalization_params)

print("Normalized Partial Points:\n", normalized_partial_points)