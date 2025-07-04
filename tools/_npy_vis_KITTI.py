import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 1052 21  1284
pcd_dir_05 = '../inference_result_PCA_05zfps_rand/324434f8eea2839bf63ee8a34069b7c5/10' # SeqPoinTr
pcd_dir_ormr = '../inference_result_PCA_originrr_aug_ws09r_KITTI/Car_3_Track_1' # AdaPoinTr - baseline
pcd_dir_05aug = '../inference_result_PCA_zfps_augrr_KITTI/Car_3_Track_1' # aug SeqPoinTr

pcd_dir_p = '../data/KITTI/Car_1_Track_0/' # input
files_p = sorted([f for f in os.listdir(pcd_dir_p) if f.endswith('.pcd')])
point_sequences_p = []
for file in files_p:
    pcd_path = os.path.join(pcd_dir_p, file)
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)         
    point_sequences_p.append(points)       

files_05 = sorted([f for f in os.listdir(pcd_dir_05) if f.endswith('_fine.npy')])
point_sequences_05 = [np.load(os.path.join(pcd_dir_05, file)) for file in files_05]

files_ormr = sorted([f for f in os.listdir(pcd_dir_ormr) if f.endswith('_fine.npy')])
point_sequences_ormr = [np.load(os.path.join(pcd_dir_ormr, file)) for file in files_ormr]

files_05aug = sorted([f for f in os.listdir(pcd_dir_05aug) if f.endswith('_fine.npy')])
point_sequences_05aug = [np.load(os.path.join(pcd_dir_05aug, file)) for file in files_05aug]

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

# ani = FuncAnimation(fig, update, frames=len(point_sequences_ormr), interval=200)
# plt.show()

# ax.set_title(f"Frame gt")
ax.axis('off')
points_05 = point_sequences_05[0]
points_ormr = point_sequences_ormr[0]
points_05aug = point_sequences_05aug[0]
points_p = point_sequences_p[0]
# ax.scatter(points_05[:, 2], points_05[:, 0], points_05[:, 1], c='blue', alpha=1, s=0.1)
# ax.scatter(points_05aug[:, 2], points_05aug[:, 0], points_05aug[:, 1], c='blue', alpha=1, s=0.1)
# ax.scatter(points_ormr[:, 2], points_ormr[:, 0], points_ormr[:, 1], c='blue', alpha=1, s=0.1)
ax.scatter(points_p[:, 2], points_p[:, 0], points_p[:, 1], c='red', alpha=1, s=0.1)
# ax.scatter(pcd_gt[:, 2], pcd_gt[:, 0], pcd_gt[:, 1], c='red', alpha=1, s=0.1)
plt.show()