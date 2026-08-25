import open3d as o3d
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import pdb; pdb.set_trace()
if o3d.core.cuda.is_available():
    print("CUDA is available. GPU is being used.")
else:
    print("CUDA is not available. CPU is being used.")

# 生成随机的点云数据
num_points = 1000
xyz = np.random.rand(num_points, 3)  # 随机生成1000个点的坐标，范围在0到1之间
rgb = np.random.rand(num_points, 3)  # 随机生成1000个点的颜色，范围在0到1之间

# 创建Open3D点云对象
point_cloud = o3d.geometry.PointCloud()

# 将生成的点云数据赋值给Open3D的点云对象
point_cloud.points = o3d.utility.Vector3dVector(xyz)
point_cloud.colors = o3d.utility.Vector3dVector(rgb)

# 渲染点云
o3d.visualization.draw_geometries([point_cloud], window_name="Random Point Cloud")

# 也可以保存点云到文件
# o3d.io.write_point_cloud("random_point_cloud.ply", point_cloud)