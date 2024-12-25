import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


pcd_dir = '../inference_result_PCA_05zfpsr/10555502fa7b3027283ffcfc40c29975/00'
files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('_fine.npy')])
point_sequences = [np.load(os.path.join(pcd_dir, file)) for file in files]


# Set equal aspect ratio for 3D
def set_equal_aspect(ax, points):
    max_range = np.ptp(points, axis=0).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


# Initialize the plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Set initial view and axis limits
points = point_sequences[0]
set_equal_aspect(ax, points)
ax.scatter(points[:, 2], points[:, 0], points[:, 1], c='blue', alpha=0.6, s=0.1)
ax.set_title("Frame 0")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=45)  # Set isometric view for the first frame


# Update function for animation
def update(frame):
    points = point_sequences[frame]
    # Remove the previous points by removing the scatter plot object
    for artist in ax.artists + ax.collections:
        artist.remove()
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], c='blue', alpha=0.6, s=0.1)
    set_equal_aspect(ax, points)
    ax.set_title(f"Frame {frame}")


# Animate the sequence
ani = FuncAnimation(fig, update, frames=len(point_sequences), interval=200)

# Show the animation
plt.show()
        # o3d.visualization.draw_geometries([pcd])

# pcd_list = []
# pcd5_list = []
# for file in sorted(os.listdir(pcd_dir)):
#     points = np.load(os.path.join(pcd_dir, file, 'fine.npy'))
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.paint_uniform_color([0, 0, 1])
#     # pcd.translate([-0.5, 0, 0])
#
#     points6 = np.load(os.path.join(pcd_dir, file, 'gt.npy'))
#     pcd6 = o3d.geometry.PointCloud()
#     pcd6.points = o3d.utility.Vector3dVector(points6)
#     pcd6.paint_uniform_color([0, 1, 0])
#     pcd6.translate([0, 0, 0])
#
#     # Step 1: Compute distances from each point in pcd to its nearest neighbor in pcd6
#     distances = np.asarray(pcd.compute_point_cloud_distance(pcd6))
#     # Step 2: Define a distance threshold
#     distance_threshold = 0.01  # Set this to the desired value
#     # Step 3: Filter out points in pcd that are beyond the distance threshold
#     mask = distances > distance_threshold
#     filtered_points = np.asarray(pcd.points)[mask]
#     # Create a new point cloud with only the filtered points
#     pcd_filtered = o3d.geometry.PointCloud()
#     pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
#     pcd_filtered.paint_uniform_color([0, 0, 1])
#     pcd_list.append(len(pcd_filtered.points))
#
#
#
#     # points2 = np.load(os.path.join(pcd_dir, file, 'gt.npy'))
#     # pcd2 = o3d.geometry.PointCloud()
#     # pcd2.points = o3d.utility.Vector3dVector(points2)
#     # pcd2.paint_uniform_color([0, 1, 0])
#
#     points5 = np.load(os.path.join(pcd_dir2, file, 'fine.npy'))
#     pcd5 = o3d.geometry.PointCloud()
#     pcd5.points = o3d.utility.Vector3dVector(points5)
#     pcd5.paint_uniform_color([1, 0, 0])
#     # pcd5.translate([0.5, 0, 0])
#
#     # Step 1: Compute distances from each point in pcd to its nearest neighbor in pcd6
#     distances = np.asarray(pcd5.compute_point_cloud_distance(pcd6))
#     # Step 2: Define a distance threshold
#     distance_threshold = 0.01  # Set this to the desired value
#     # Step 3: Filter out points in pcd that are beyond the distance threshold
#     mask = distances > distance_threshold
#     filtered_points5 = np.asarray(pcd5.points)[mask]
#     # Create a new point cloud with only the filtered points
#     pcd5_filtered = o3d.geometry.PointCloud()
#     pcd5_filtered.points = o3d.utility.Vector3dVector(filtered_points5)
#     pcd5_filtered.paint_uniform_color([1, 0, 0])
#     pcd5_filtered.translate([0.5, 0, 0])
#     pcd5_list.append(len(pcd5_filtered.points))
#
#     # points7 = np.load(os.path.join(pcd_dir, file, 'gt.npy'))
#     # pcd7 = o3d.geometry.PointCloud()
#     # pcd7.points = o3d.utility.Vector3dVector(points7)
#     # pcd7.paint_uniform_color([0, 1, 0])
#     # pcd7.translate([0.5, 0, 0])
#
#
#     # pcd3 = o3d.geometry.PointCloud()
#     # pcd3.points = o3d.utility.Vector3dVector(points)
#     # pcd3.paint_uniform_color([0, 0, 1])
#     # pcd3.translate([0.5, 0, 0])
#     #
#     # pcd4 = o3d.geometry.PointCloud()
#     # pcd4.points = o3d.utility.Vector3dVector(points2)
#     # pcd4.paint_uniform_color([0, 1, 0])
#     # pcd4.translate([0.5, 0, 0])
#
#     # pcd.paint_uniform_color([0, 0, 1])
#     #o3d.visualization.draw_geometries([pcd_filtered, pcd5_filtered])
#
# print(np.mean(pcd_list))
# print(np.mean(pcd5_list))