import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


pcd_dir_05 = '../inference_result_PCA_05zfpsr/324434f8eea2839bf63ee8a34069b7c5/10'
pcd_dir_or = '../inference_result_PCA_originrr/324434f8eea2839bf63ee8a34069b7c5/10'
pcd_dir_ormr = '../inference_result_PCA_originrr_aug_ws09r_LD/Car_111'
pcd_gt_path = '../data/ShapeNet_Car_Seq/test/complete/3373140534463359fc82e75321e09f82.pcd'
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

files_ormr = sorted([f for f in os.listdir(pcd_dir_ormr) if f.endswith('_fine.npy')])
point_sequences_ormr = [np.load(os.path.join(pcd_dir_ormr, file)) for file in files_ormr]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Set equal aspect ratio for 3D
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)

ax.axis('off')
ax.view_init(elev=0, azim=-90)
# ax.view_init(elev=30, azim=-45)  # Set isometric view
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

ani = FuncAnimation(fig, update, frames=len(point_sequences_ormr), interval=200)
plt.show()

# ax.set_title(f"Frame gt")
# ax.axis('off')
# points_05 = point_sequences_05[39]
# points_or = point_sequences_or[43]
# points_ormr = point_sequences_ormr[39]
# points_p = point_sequences_p[41]
# # ax.scatter(points_05[:, 2], points_05[:, 0], points_05[:, 1], c='blue', alpha=1, s=0.1)
# # ax.scatter(points_or[:, 2], points_or[:, 0], points_or[:, 1], c='blue', alpha=1, s=0.1)
# # ax.scatter(points_ormr[:, 2], points_ormr[:, 0], points_ormr[:, 1], c='blue', alpha=1, s=0.1)
# # ax.scatter(points_p[:, 2], points_p[:, 0], points_p[:, 1], c='red', alpha=1, s=0.1)
# ax.scatter(pcd_gt[:, 2], pcd_gt[:, 0], pcd_gt[:, 1], c='blue', alpha=1, s=0.1)
# plt.show()