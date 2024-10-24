import numpy as np
import open3d as o3d
import os


pcd_dir = '../inference_result_PCA_00'
pcd_dir2 = '../inference_result_PCA_02'

pcd_list = []
pcd5_list = []
for file in sorted(os.listdir(pcd_dir)):
    points = np.load(os.path.join(pcd_dir, file, 'fine.npy'))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    # pcd.translate([-0.5, 0, 0])

    points6 = np.load(os.path.join(pcd_dir, file, 'gt.npy'))
    pcd6 = o3d.geometry.PointCloud()
    pcd6.points = o3d.utility.Vector3dVector(points6)
    pcd6.paint_uniform_color([0, 1, 0])
    pcd6.translate([0, 0, 0])

    # Step 1: Compute distances from each point in pcd to its nearest neighbor in pcd6
    distances = np.asarray(pcd.compute_point_cloud_distance(pcd6))
    # Step 2: Define a distance threshold
    distance_threshold = 0.01  # Set this to the desired value
    # Step 3: Filter out points in pcd that are beyond the distance threshold
    mask = distances > distance_threshold
    filtered_points = np.asarray(pcd.points)[mask]
    # Create a new point cloud with only the filtered points
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
    pcd_filtered.paint_uniform_color([0, 0, 1])
    pcd_list.append(len(pcd_filtered.points))



    # points2 = np.load(os.path.join(pcd_dir, file, 'gt.npy'))
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(points2)
    # pcd2.paint_uniform_color([0, 1, 0])

    points5 = np.load(os.path.join(pcd_dir2, file, 'fine.npy'))
    pcd5 = o3d.geometry.PointCloud()
    pcd5.points = o3d.utility.Vector3dVector(points5)
    pcd5.paint_uniform_color([1, 0, 0])
    # pcd5.translate([0.5, 0, 0])

    # Step 1: Compute distances from each point in pcd to its nearest neighbor in pcd6
    distances = np.asarray(pcd5.compute_point_cloud_distance(pcd6))
    # Step 2: Define a distance threshold
    distance_threshold = 0.01  # Set this to the desired value
    # Step 3: Filter out points in pcd that are beyond the distance threshold
    mask = distances > distance_threshold
    filtered_points5 = np.asarray(pcd5.points)[mask]
    # Create a new point cloud with only the filtered points
    pcd5_filtered = o3d.geometry.PointCloud()
    pcd5_filtered.points = o3d.utility.Vector3dVector(filtered_points5)
    pcd5_filtered.paint_uniform_color([1, 0, 0])
    pcd5_filtered.translate([0.5, 0, 0])
    pcd5_list.append(len(pcd5_filtered.points))

    # points7 = np.load(os.path.join(pcd_dir, file, 'gt.npy'))
    # pcd7 = o3d.geometry.PointCloud()
    # pcd7.points = o3d.utility.Vector3dVector(points7)
    # pcd7.paint_uniform_color([0, 1, 0])
    # pcd7.translate([0.5, 0, 0])


    # pcd3 = o3d.geometry.PointCloud()
    # pcd3.points = o3d.utility.Vector3dVector(points)
    # pcd3.paint_uniform_color([0, 0, 1])
    # pcd3.translate([0.5, 0, 0])
    #
    # pcd4 = o3d.geometry.PointCloud()
    # pcd4.points = o3d.utility.Vector3dVector(points2)
    # pcd4.paint_uniform_color([0, 1, 0])
    # pcd4.translate([0.5, 0, 0])

    # pcd.paint_uniform_color([0, 0, 1])
    #o3d.visualization.draw_geometries([pcd_filtered, pcd5_filtered])

print(np.mean(pcd_list))
print(np.mean(pcd5_list))