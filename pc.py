import numpy as np
import cv2
import open3d as o3d
from PIL import Image



def get_pointcloud(cpath,dpath,k,depth_threshold):

    # pointcloud = get_pointcloud("E://img_//color_img4_out//d30.png", "E://img_//depth_img4_out//d30.png", k)
    image = np.array(Image.open(cpath))
    # plt.imshow(image, cmap='gray')
    # plt.show()
    # 读取深度
    Zc = np.array(Image.open(dpath)).astype(np.float32)
    # 去除某些离群点
    # invalid_mask = Zc > 180
    # Zc[invalid_mask]  = 0
    # 将深度图中超过阈值的部分置为0
    depth_mask = Zc <= depth_threshold
    Zc[~depth_mask] = 0

    cx = k[0][2]
    cy = k[1][2]
    fx = k[0][0]
    fy = k[1][1]

    p1_v = np.reshape(np.linspace(0, Zc.shape[0] - 1, Zc.shape[0]), (-1, 1)).repeat(axis=1, repeats=Zc.shape[1]) # 列向量复制成矩阵
    p1_u = np.reshape(np.linspace(0, Zc.shape[1] - 1, Zc.shape[1]), (1, -1)).repeat(axis=0, repeats=Zc.shape[0]) # 行向量复制成矩阵
    xc = Zc * (p1_u - cx) / fx  # 3D点的横坐标矩阵
    yc = Zc * (p1_v - cy) / fy  # 3D点的纵坐标矩阵

    # 使用 np.expand_dims() 函数在 xc 矩阵的第三个维度上添加一个额外的维度。这样做是为了将 xc 从二维矩阵变为三维矩阵，以便后续的拼接操作。
    # (480,640,1)
    xc = np.expand_dims(xc, axis=2)
    yc = np.expand_dims(yc, axis=2)
    Zc = np.expand_dims(Zc, axis=2)
    # np.concatenate() 函数在第三个维度上将 xc、yc 和 Zc 进行拼接，生成一个新的三维矩阵 xyz
    # (480,640,3)
    xyz = np.concatenate((xc, yc, Zc), axis=2)
    # (307200,3)
    xyz = np.reshape(xyz, (-1, 3)).astype(np.float32)
    rgb = np.reshape(image, (-1, 3)).astype(np.float32)
    rgb = rgb / 255

    pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(xyz))
    pcd.colors = o3d.pybind.utility.Vector3dVector(rgb)
    # o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=False)
    # xyz
    return pcd



k1 = np.array([[516.97781372070313, 0.000000, 306.94604492187500 ],[0.000000 ,516.97781372070313 ,257.43075942993164],[0.000000, 0.000000 ,1.000000 ]])

# k2 = np.array([[609.998028, 0.0, 323.180504], [0.0, 613.391791, 257.701951], [0.0, 0.0, 1.0]])
# k3 = np.array([[602.126810, 0.0, 323.977947], [0.0, 602.467833, 244.198967], [0.0, 0.0, 1.0]])
# k4 = np.array([[609.998028, 0.0, 323.180504], [0.0, 613.391791, 257.701951], [0.0, 0.0, 1.0]])

cpath1 = r"D:\try\try\MIOL\output_frames\rect_left_gai\01846.jpg"

# cpath2 = "E://try_img//img_man//color_img2_out//c30.png"
# cpath3 = "E://try_img//img_man//color_img3_out//c30.png"
# cpath4 = "E://try_img//img_man//color_img4_out//c30.png"

dpath1 = r"D:\try\try\MIOL\output_frames_1105_dep\show_dep001846.png"

# dpath2 = "E://try_img//img_man//depth_img2_out//d30.png"
# dpath3 = "E://try_img//img_man//depth_img3_out//d30.png"
# dpath4 = "E://try_img//img_man//depth_img4_out//d30.png"

pcd1 = get_pointcloud(cpath1,dpath1,k1,1600)

# pcd2 = get_pointcloud(cpath2,dpath2,k2,1600)
# pcd3 = get_pointcloud(cpath3,dpath3,k3,1600)
# pcd4 = get_pointcloud(cpath4,dpath4,k4,1600)

# t1 = np.array([[ 0.318325 , -0.578911 ,  0.750687,-388.933], [-0.943149 , -0.113548 ,  0.312371,  -129.281], [-0.0955962 , -0.807446 , -0.582145, 509.272],[0,0,0,1]])
#
# t2 = np.array([[-0.411845 , 0.545121, -0.730224,942.094], [0.899556 , 0.115223 ,-0.421334,647.606], [-0.14554 ,-0.830402 ,-0.537821,520.183],[0,0,0,1]])
# t3 = np.array([[0.542615 , 0.426049 ,-0.723914,897.932], [0.833979 ,-0.376106 , 0.403763, -214.015], [-0.100246 ,-0.822817 ,-0.559396, 520.115],[0,0,0,1]])
# t4 = np.array([[-0.637319 ,-0.265011 , 0.723598,-390.675], [ -0.757932 , 0.385138, -0.526506,665.023], [-0.139156,  -0.88399, -0.446316,513.197],[0,0,0,1]])

tz = np.eye(4)
tz[0][0]=-1
tz[1][1]=-1

# pcd11 =pcd1.transform(t1)
# pcd22 =pcd2.transform(t2)
# pcd33 =pcd3.transform(t3)
# pcd44 =pcd4.transform(t4)
# pcdall = pcd11 + pcd22 + pcd33 + pcd44

# def custom_draw_geometry_with_rotation(pcd):
#
#     def rotate_view(vis):
#         ctr = vis.get_view_control()
#         ctr.rotate(10.0, 10.0)
#         return False
#
#     o3d.visualization.draw_geometries_with_animation_callback([pcdall],
#                                                               rotate_view,window_name='Open3D', width=1920, height=1080, left=50, top=50)
#
# custom_draw_geometry_with_rotation(pcdall)

o3d.visualization.draw_geometries([pcd1])

