import open3d as o3d
import os

file = r"D:\Ji\PoinTr\data\ShapeNet_Car_Seq\test\complete\12243301d1c8148e33d7c9e122eec9b6.pcd"

if os.path.exists(file):
    print("File found.")
    pcd = o3d.io.read_point_cloud(file)
    o3d.visualization.draw_geometries([pcd])
else:
    print("File not found.")