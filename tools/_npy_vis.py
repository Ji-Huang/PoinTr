import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


pcd_dir_05 = '../inference_result_PCA_05zfpsr/324434f8eea2839bf63ee8a34069b7c5/10'
pcd_dir_or = '../inference_result_PCA_originrr/324434f8eea2839bf63ee8a34069b7c5/10'
pcd_dir_orm = '../inference_result_PCA_originrr_ws09/10555502fa7b3027283ffcfc40c29975/00/'
pcd_dir_ormr = '../inference_result_PCA_originrr_ws09r/324434f8eea2839bf63ee8a34069b7c5/10'
pcd_gt_path = '../data/ShapeNet_Car_Seq/test/complete/324434f8eea2839bf63ee8a34069b7c5.pcd'
pcd_dir_p = '../data/ShapeNet_Car_Seq/test/partial/272791fdabf46b2d5921daf0138cfe67/02'
pcd_gt = o3d.io.read_point_cloud(pcd_gt_path)
pcd_gt = np.asarray(pcd_gt.points)
# pcd_dir = '../inference_result_PCA_originrr/10555502fa7b3027283ffcfc40c29975/00'
# pcd_dir = '../inference_result_PCA_05zfpsr/Car_111'
files_p = sorted([f for f in os.listdir(pcd_dir_p) if f.endswith('.pcd')])
point_sequences_p = []
for file in files_p:
    pcd_path = os.path.join(pcd_dir_p, file)
    pcd = o3d.io.read_point_cloud(pcd_path)  # Read the PCD file
    points = np.asarray(pcd.points)         # Convert to NumPy array
    point_sequences_p.append(points)        # Store in the list

# point_sequences_p = [np.load(os.path.join(pcd_dir_p, file)) for file in files_p]

files_05 = sorted([f for f in os.listdir(pcd_dir_05) if f.endswith('_fine.npy')])
point_sequences_05 = [np.load(os.path.join(pcd_dir_05, file)) for file in files_05]

files_or = sorted([f for f in os.listdir(pcd_dir_or) if f.endswith('_fine.npy')])
point_sequences_or = [np.load(os.path.join(pcd_dir_or, file)) for file in files_or]

files_orm = sorted([f for f in os.listdir(pcd_dir_orm) if f.endswith('_fine.npy')])
point_sequences_orm = [np.load(os.path.join(pcd_dir_orm, file)) for file in files_orm]

files_ormr = sorted([f for f in os.listdir(pcd_dir_ormr) if f.endswith('_fine.npy')])
point_sequences_ormr = [np.load(os.path.join(pcd_dir_ormr, file)) for file in files_ormr]
# point_sequences_or = point_sequences_or[4:]
# point_sequences_or = point_sequences_or[:-4]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Set equal aspect ratio for 3D
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)

# Set initial view and axis limits
# ax.scatter(points[:, 2], points[:, 0], points[:, 1], c='blue', alpha=0.6, s=0.1)
ax.axis('off')
ax.view_init(elev=0, azim=-90)
# ax.view_init(elev=30, azim=-45)  # Set isometric view for the first frame
# ax.view_init(elev=90, azim=-90)
# Update function for animation
def update(frame):
    points = point_sequences_ormr[frame]
    # Remove the previous points by removing the scatter plot object
    for artist in ax.artists + ax.collections:
        artist.remove()
    count = int(frame) #+ 4
    ax.set_title(f"Frame {count}")
    ax.axis('off')
    ax.scatter(points[:, 2], points[:, 0], points[:, 1], c='blue', alpha=1, s=0.1)

# for i, frame in enumerate(point_sequences):
#     update(i)  # Draw the current frame
#     # plt.show()
#     path = 'D:/Ji/PoinTr/fig/PCA_originrr/324434f8eea2839bf63ee8a34069b7c5/10/'
#     os.makedirs(path, exist_ok=True)
#     plt.savefig(os.path.join(path, f'frame_{frame:03}.png'), dpi=300, bbox_inches='tight', pad_inches=0)  # Save with a sequential filename

# ani = FuncAnimation(fig, update, frames=len(point_sequences_or), interval=200)
# plt.show()

# ax.set_title(f"Frame gt")
ax.axis('off')
points_05 = point_sequences_05[39]
points_or = point_sequences_or[43]
points_orm = point_sequences_orm[10]
points_ormr = point_sequences_ormr[39]
points_p = point_sequences_p[41]
ax.scatter(points_05[:, 2], points_05[:, 0], points_05[:, 1], c='blue', alpha=1, s=0.1)
# ax.scatter(points_or[:, 2], points_or[:, 0], points_or[:, 1], c='blue', alpha=1, s=0.1)
# ax.scatter(points_orm[:, 2], points_orm[:, 0], points_orm[:, 1], c='blue', alpha=1, s=0.1)
# ax.scatter(points_ormr[:, 2], points_ormr[:, 0], points_ormr[:, 1], c='blue', alpha=1, s=0.1)
# ax.scatter(points_p[:, 2], points_p[:, 0], points_p[:, 1], c='red', alpha=1, s=0.1)
# ax.scatter(pcd_gt[:, 2], pcd_gt[:, 0], pcd_gt[:, 1], c='blue', alpha=1, s=0.1)
plt.show()
# 10555502fa7b3027283ffcfc40c29975/02/ 25 29  # 00 10 14
# 12097984d9c51437b84d944e8a1952a5/06/ 31 35
# 202648a87dd6ad2573e10a7135e947fe/06/ 31 35
# 272791fdabf46b2d5921daf0138cfe67/02 33 37
# 324434f8eea2839bf63ee8a34069b7c5/10/ 35-43 39-47

# plt.close(fig)  # Close the figure when done


# pcd_dir = '../inference_result_PCA_00'
# pcd_dir2 = '../inference_result_PCA_02o'
#
# pcd_list = []
# pcd5_list = []
# for file in sorted(os.listdir(pcd_dir)):
#     points = np.load(os.path.join(pcd_dir, file, 'fine.npy'))
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.paint_uniform_color([0, 0, 1])
#     pcd.translate([-0.5, 0, 0])
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
#     pcd5.translate([0.5, 0, 0])
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
#     o3d.visualization.draw_geometries([pcd, pcd5, pcd6])

# print(np.mean(pcd_list))
# print(np.mean(pcd5_list))