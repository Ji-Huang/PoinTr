import numpy as np
import open3d as o3d
import os


pcd_dir = '../inference_result_ShapeNet_Car_04'
for file in sorted(os.listdir(pcd_dir)):
    points = np.load(os.path.join(pcd_dir, file, 'fine.npy'))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd])