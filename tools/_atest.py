import numpy as np
import open3d as o3d
# import pypcd
import os
import random
import shutil
import json
import time

from scipy.spatial.distance import cdist
from random import randint


# PoinTr:
# input: 2048 -> embedding: 256 -> output: 16384
# GT: 8192

# PCN:
# input: 1024 -> coarse output: 1024 -> dense output: 16384
# GT: coarse 1024, dense 16384

# OURS:
# input: 256 -> output: 16384


def check_traj_pcd_length():
    save_dir = "/users/jihuang/SUSTechPOINTS/data/0825/"

    len_dict = {}
    for pcd_file in sorted(os.listdir(save_dir)):
        len_dict[pcd_file] = {}
        for traj_file in sorted(os.listdir(os.path.join(save_dir, pcd_file))):
            len_dict[pcd_file][traj_file] = []
            for mesh_file in sorted(os.listdir(os.path.join(save_dir, pcd_file, traj_file))):
                len_dict[pcd_file][traj_file].append(
                    len(sorted(os.listdir(os.path.join(save_dir, pcd_file, traj_file, mesh_file)))))
    # print(len_dict)

    # for pcd_file, traj_files in len_dict.items():
    #     print(f"{pcd_file}:")
    #     for traj_file, lengths in traj_files.items():
    #         print(f"  {traj_file}: {lengths}")

    for pcd_file, traj_files in len_dict.items():
        print(f"{pcd_file}:")
        for traj_file, lengths in traj_files.items():
            min_len = min(lengths)
            max_len = max(lengths)
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            if min_len < 21:
                print(f"  {traj_file}: Min: {min_len}, Max: {max_len}, Avg: {avg_len}")


# plt.figure(figsize=(10, 6))
# plt.plot(spline_points[:, 0], spline_points[:, 1], label='Catmull-Rom Spline Trajectory')
# waypoints_s = np.array(waypoints[0])
# plt.scatter(waypoints_s[:, 0], waypoints_s[:, 1], color='red', label='Waypoints')
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Catmull-Rom Spline Trajectory')
# plt.grid(True)
# plt.show()


def adaPT_data_format_PCN():
    save_dir = "/users/jihuang/SUSTechPOINTS/data/0825/"
    save_dir_new = "/users/jihuang/SUSTechPOINTS/data/0828/"

    for pcd_file in sorted(os.listdir(save_dir)):
        for traj_file in sorted(os.listdir(os.path.join(save_dir, pcd_file))):
            for mesh_file in sorted(os.listdir(os.path.join(save_dir, pcd_file, traj_file))):
                pcd_list = sorted(os.listdir(os.path.join(save_dir, pcd_file, traj_file, mesh_file)))

                # Randomly select 8 pcd files
                selected_pcds = random.sample(pcd_list, 8)

                # Create new directory for selected pcd files
                # new_dir = os.path.join(save_dir_new, pcd_file, traj_file, mesh_file)
                new_dir = os.path.join(save_dir_new, f"{pcd_file}_{traj_file}", mesh_file)
                os.makedirs(new_dir, exist_ok=True)

                # Copy selected pcd files to new directory and rename them
                for i, pcd in enumerate(selected_pcds):
                    old_path = os.path.join(save_dir, pcd_file, traj_file, mesh_file, pcd)
                    new_path = os.path.join(new_dir, f"{str(i).zfill(2)}.pcd")
                    shutil.copyfile(old_path, new_path)


def data_json_gen():
    dir_path = "/users/jihuang/SUSTechPOINTS/data/0828/"

    data = []

    for taxonomy_id in sorted(os.listdir(dir_path)):
        taxonomy_path = os.path.join(dir_path, taxonomy_id)

        # Check if it's a directory
        if os.path.isdir(taxonomy_path):
            taxonomy_name = f"n_{taxonomy_id}"

            # Get the subdirectories in the current directory
            subdirs = sorted([d for d in os.listdir(taxonomy_path) if os.path.isdir(os.path.join(taxonomy_path, d))])

            # Divide the subdirectories into three categories
            test_files = subdirs[:4]
            train_files = subdirs[4:56]
            val_files = subdirs[56:]

            # Create a dictionary for the subdirectory
            subdir_dict = {
                "taxonomy_id": taxonomy_id,
                "taxonomy_name": taxonomy_name,
                "test": test_files,
                "train": train_files,
                "val": val_files
            }

            # Append the dictionary to the list
            data.append(subdir_dict)

    # Write the list to a JSON file
    with open('/users/jihuang/PCN.json', 'w') as f:
        json.dump(data, f, indent=4)


def dir_move():
    # Define the directories to be moved
    dirs_to_move = ["xiandai-i25-2016", "yingfeinidi-qx80", "yiqi-benteng-b50"]

    # Define the source and destination directories
    source_dir = "/users/jihuang/PCN/train/partial/"
    dest_dir = "/users/jihuang/PCN/val/partial/"
    os.makedirs(dest_dir, exist_ok=True)

    # Loop over the directories in the source directory
    for taxonomy_id in os.listdir(source_dir):
        taxonomy_source_dir = os.path.join(source_dir, taxonomy_id)
        taxonomy_dest_dir = os.path.join(dest_dir, taxonomy_id)

        # Loop over the directories in the taxonomy directory
        for dir_name in os.listdir(taxonomy_source_dir):
            # If the directory is in the list of directories to be moved
            if dir_name in dirs_to_move:
                # Construct the source and destination paths
                source_path = os.path.join(taxonomy_source_dir, dir_name)
                dest_path = os.path.join(taxonomy_dest_dir, dir_name)

                # Move the directory
                shutil.move(source_path, dest_path)


def create_subdir_copy_pcd():
    source_dir = "/users/jihuang/PCN/val/partial/"
    dest_dir = "/users/jihuang/PCN/val/complete/"
    pcd_dir = "/users/jihuang/x_gt_s/val"

    for dir_name in os.listdir(source_dir):
        new_dir_path = os.path.join(dest_dir, dir_name)
        os.makedirs(new_dir_path, exist_ok=True)

        for file_name in os.listdir(pcd_dir):
            shutil.copy(os.path.join(pcd_dir, file_name), new_dir_path)


def create_txt():
    json_path = "/users/jihuang/PCN/PCN.json"

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract the "taxonomy_name" values
    taxonomy_names = [item["taxonomy_name"] for item in data]

    # Write the "taxonomy_name" values to a text file
    with open("/users/jihuang/PCN/category.txt", 'w') as f:
        for name in taxonomy_names:
            f.write(name + '\n')


def farthest_point_sampling(points, num_samples):
    farthest_pts = np.zeros((num_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.full(len(points), np.inf)
    for i in range(1, num_samples):
        distances = np.minimum(distances, np.sum((points - farthest_pts[i - 1]) ** 2, axis=-1))
        farthest_pts[i] = points[np.argmax(distances)]
    return farthest_pts


def reduce_to_256():
    save_dir = "/users/jihuang/PCN/val/partial/"

    for traj_file in sorted(os.listdir(save_dir)):
        for mesh_file in sorted(os.listdir(os.path.join(save_dir, traj_file))):
            for pcd_file in sorted(os.listdir(os.path.join(save_dir, traj_file, mesh_file))):
                pc = pypcd.PointCloud.from_path(os.path.join(save_dir, traj_file, mesh_file, pcd_file))
                points = np.stack([pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z']], axis=-1)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(farthest_point_sampling(points, 256))

                # Create the new directory if it doesn't exist
                new_dir = os.path.join("/users/jihuang/PCN/val/reduced/", traj_file, mesh_file)
                os.makedirs(new_dir, exist_ok=True)

                # Save the reduced point cloud to the new directory
                o3d.io.write_point_cloud(os.path.join(new_dir, pcd_file), pcd)


def add_points_by_interpolation(pcd):
    """
    Add points into a point cloud by interpolating between random pairs of close points.
    """
    points = np.asarray(pcd.points)
    new_points = []

    for _ in range(16384 - len(points)):
        # Compute pairwise distances
        distances = cdist(points, points)

        # Find the closest points for each point
        closest_points = np.argsort(distances, axis=1)[:, 1]

        # Randomly select a pair of close points
        idx1 = randint(0, len(points) - 1)
        idx2 = closest_points[idx1]

        # Interpolate a new point between the selected pair
        t = np.random.uniform(0, 1)
        new_point = (1 - t) * points[idx1] + t * points[idx2]

        new_points.append(new_point)

    # Add the new points to the point cloud
    all_points = np.concatenate([points, new_points])

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(all_points)

    # Visualize the point clouds
    # pcd_original = o3d.geometry.PointCloud()
    # pcd_original.points = o3d.utility.Vector3dVector(points)
    #
    # pcd_new = o3d.geometry.PointCloud()
    # pcd_new.points = o3d.utility.Vector3dVector(new_points)
    #
    # pcd_original.paint_uniform_color([1, 0, 0])  # Red color for original points
    # pcd_new.paint_uniform_color([0, 1, 0])  # Green color for new points
    #
    # o3d.visualization.draw_geometries([pcd_original, pcd_new])

    return new_pcd


def check_pcd_number():
    save_dir = "/users/jihuang/PCN/train/complete/1_0619_00"

    for pcd_file in sorted(os.listdir(save_dir)):
        pc = pypcd.PointCloud.from_path(os.path.join(save_dir, pcd_file))
        points = np.stack([pc.pc_data['x'], pc.pc_data['y'], pc.pc_data['z']], axis=-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # print(len(pcd.points))

        if len(pcd.points) < 16384:
            print(len(pcd.points))
            new_pcd = add_points_by_interpolation(pcd)

            print("to:", len(new_pcd.points))
            o3d.io.write_point_cloud(os.path.join("/users/jihuang/PCN/complete16384/", pcd_file), new_pcd)


def substitute_files():
    # Define the source and target directories
    source_dir = "D:/ji/PoinTr/data/x_gt_s_p/"
    target_dir = "D:/ji/PoinTr/data/PCN/val/complete/"

    # Iterate over all files in the target directory
    for traj_file in sorted(os.listdir(target_dir)):
        for pcd_file in sorted(os.listdir(os.path.join(target_dir, traj_file))):
            # Check if the file is a .pcd file
            if pcd_file.endswith('.pcd'):
                # Construct the full paths to the source and target files
                source_file = os.path.join(source_dir, pcd_file)
                target_file = os.path.join(target_dir, traj_file, pcd_file)

                # Check if the source file exists
                if os.path.exists(source_file):
                    print(f"Replacing {target_file} with {source_file}")
                    # Replace the target file with the source file
                    shutil.copyfile(source_file, target_file)


def PCN2LD():
    # PCN_dir = "../data/PCN_o/test/complete"
    # PCN_dir2 = "../data/PCN_t/test/complete"
    # for traj_file in sorted(os.listdir(PCN_dir)):
    #     for pcd_file in sorted(os.listdir(os.path.join(PCN_dir, traj_file))):
    #         if pcd_file.endswith('.pcd'):
    #             pcd_PCN = o3d.io.read_point_cloud((os.path.join(PCN_dir, traj_file, pcd_file)))
    #             pcd_PCN.rotate(pcd_PCN.get_rotation_matrix_from_xyz([0, -np.pi / 2, -np.pi / 2]),
    #                            center=pcd_PCN.get_center())
    #             pcd_PCN.scale(6, center=pcd_PCN.get_center())
    #
    #             new_dir = os.path.join(PCN_dir2, traj_file)
    #             os.makedirs(new_dir, exist_ok=True)
    #             o3d.io.write_point_cloud(os.path.join(new_dir, pcd_file), pcd_PCN)
    PCN_dir = "../data/PCN_o/test/partial"
    PCN_dir2 = "../data/PCN_t/test/partial"
    for traj_file in sorted(os.listdir(PCN_dir)):
        for pcd_file in sorted(os.listdir(os.path.join(PCN_dir, traj_file))):
            for pcd in sorted(os.listdir(os.path.join(PCN_dir, traj_file, pcd_file))):
                if pcd.endswith('.pcd'):
                    pcd_PCN = o3d.io.read_point_cloud((os.path.join(PCN_dir, traj_file, pcd_file, pcd)))
                    pcd_PCN.rotate(pcd_PCN.get_rotation_matrix_from_xyz([0, -np.pi / 2, -np.pi / 2]),
                                   center=pcd_PCN.get_center())
                    pcd_PCN.scale(6, center=pcd_PCN.get_center())

                    new_dir = os.path.join(PCN_dir2, traj_file, pcd_file)
                    os.makedirs(new_dir, exist_ok=True)
                    o3d.io.write_point_cloud(os.path.join(new_dir, pcd), pcd_PCN)


def copy_pcd_files():
    src_dir = '../data/PCN/test/partial/02958343'
    dst_dir = '../demo'
    for dirs in sorted(os.listdir(src_dir)):
        for file in sorted(os.listdir(os.path.join(src_dir, dirs))):
            if file.endswith('.pcd'):
                pcd = o3d.io.read_point_cloud(os.path.join(src_dir, dirs, file))
                o3d.io.write_point_cloud(os.path.join(dst_dir, dirs + file), pcd)


def data_json_gen_shapenet():
    dir_path1 = 'D:/Ji/PoinTr/data/ShapeNet_Car_Seq/train/complete'
    dir_path2 = 'D:/Ji/PoinTr/data/ShapeNet_Car_Seq/test/complete'

    # List all files and remove the .pcd extension from each filename
    train_files = sorted([f.replace('.pcd', '') for f in os.listdir(dir_path1)])
    test_files = sorted([f.replace('.pcd', '') for f in os.listdir(dir_path2)])

    # Create a dictionary with the train and test files
    data = {
        "train": train_files,
        "test": test_files
    }

    # Write the data to a JSON file
    json_path = 'D:/Ji/PoinTr/data/ShapeNet_Car_Seq/ShapeNet_Car_Seq.json'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"JSON file saved to {json_path}")


def reformat_shapenet():
    # Define the directory path
    dir_path1 = 'D:/Ji/PoinTr/data/ShapeNet_Car_Seq/train/partial'

    # Iterate over taxonomies in the directory
    for tax in sorted(os.listdir(dir_path1)):
        tax_path = os.path.join(dir_path1, tax)
        # Iterate over trajectories in the taxonomy directory
        for traj in sorted(os.listdir(tax_path)):
            traj_path = os.path.join(tax_path, traj)

            # Get a list of all .pcd files in the trajectory directory
            pcd_files = sorted([f for f in os.listdir(traj_path) if f.endswith('.pcd')])

            # Remove 'merged.pcd' if it exists
            merged_file = os.path.join(traj_path, 'merged.pcd')
            if 'merged.pcd' in pcd_files:
                os.remove(merged_file)  # Delete the file
                pcd_files.remove('merged.pcd')  # Remove from the list

            # Now rename the remaining .pcd files in sequential order
            for idx, old_file in enumerate(pcd_files):
                new_file_name = f'{idx:03}.pcd'  # Generate a new name (e.g., 001.pcd, 002.pcd, ...)
                old_file_path = os.path.join(traj_path, old_file)
                new_file_path = os.path.join(traj_path, new_file_name)

                os.rename(old_file_path, new_file_path)  # Rename the file
            print(f'Renamed {tax}, {traj}')


def reformat_shapenet2():
    # Define the directory path
    dir_path1 = 'D:/Ji/PoinTr/LiangDao_normalized'

    # Iterate over taxonomies in the directory
    for tax in sorted(os.listdir(dir_path1)):
        tax_path = os.path.join(dir_path1, tax)

        pcd_files = sorted([f for f in os.listdir(tax_path) if f.endswith('.pcd')])
        print(pcd_files)

        # Now rename the .pcd files in sequential order
        for idx, old_file in enumerate(pcd_files):
            new_file_name = f'{idx:03}.pcd'  # Generate a new name (e.g., 001.pcd, 002.pcd, ...)
            old_file_path = os.path.join(tax_path, old_file)
            new_file_path = os.path.join(tax_path, new_file_name)

            os.rename(old_file_path, new_file_path)  # Rename the file
        print(f'Renamed {tax}, {old_file}')


def count_avg_shapenet():
    # Define the directory path
    dir_path1 = 'D:/Ji/PoinTr/data/ShapeNet_Car_Seq/train/partial/2468ceab72b4be041d9cc9d194c8533'

    # Iterate over taxonomies in the directory
    total = 0
    for traj in sorted(os.listdir(dir_path1)):
        traj_path = os.path.join(dir_path1, traj)
        # Iterate over trajectories in the taxonomy directory
        files = sorted(os.listdir(traj_path))
        total += len(files)
    print(total)
    print(total/15)

count_avg_shapenet()
# 1305/15 = 87
# 4664/47 = 99.234