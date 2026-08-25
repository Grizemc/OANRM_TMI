import numpy as np
import open3d as o3d

# 读取NPZ文件
data = np.load('rectified01_1000.npz')  # 替换'your_file.npz'为你的NPZ文件名
data1 = data['point1']
ransac_source_cloud = o3d.geometry.PointCloud(
                    o3d.pybind.utility.Vector3dVector(data1))
ransac_source_cloud.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([ransac_source_cloud])

print(0)



# # 创建可视化窗口
# vis = o3d.visualization.Visualizer()
# vis.create_window()
#
# # 添加点云到可视化窗口
# vis.add_geometry(ransac_source_cloud)
#
# # 设置相机参数以更好地查看点云（可选）
# # 例如，将相机移动到某个位置并设置其朝向
# ctr = vis.get_view_control()
# ctr.set_lookat([0, 0, 0])
# ctr.set_up([0, 0, 1])
# ctr.set_front([0, -1, 0])
#
# # 开始渲染循环
# vis.run()
#
# # 注意：在 Jupyter Notebook 或某些环境中，直接使用 vis.run() 可能不会工作。
# # 在这些情况下，你可能需要使用其他方法来渲染点云，如 o3d.visualization.draw_geometries([ransac_source_cloud])
# # 但请注意，draw_geometries() 是非阻塞的，并且不会提供一个持久的可视化窗口。