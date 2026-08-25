#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 22:42
# @Author  : 沈子明
# @File    : VisUtil.py
# @Software: PyCharm
import numpy as np
from mayavi import mlab
import open3d as o3d


def ShowTanTaiLabelPcd(points1, colors1, label_point, label_color):
    label_point[:, 2] -= 5
    pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
    pcd1.colors = o3d.pybind.utility.Vector3dVector(colors1)
    label_color[:, 0] = 0  # 绿色， 截断点云1，理论上应该完全重合点云1，且总数量少读点云1
    label_color[:, 1] = 0.5
    label_color[:, 2] = 0
    label_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(label_point))
    label_pcd.colors = o3d.pybind.utility.Vector3dVector(label_color)

    mat1 = o3d.visualization.rendering.MaterialRecord()
    # mat1.shader = 'defaultUnlit'
    mat1.point_size = 5.0  # 设置第一个点云的点大小为5.0
    label_mat = o3d.visualization.rendering.MaterialRecord()
    # mat2.shader = 'defaultUnlit'
    label_mat.point_size = 10.0  # 设置第二个点云的点大小为10.0

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d")
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)
    widget3d.scene.add_geometry("pca1", pcd1, mat1)
    widget3d.scene.add_geometry("label_pcd", label_pcd, label_mat)
    widget3d.setup_camera(60, widget3d.scene.bounding_box, [0, 0, 0])
    app.run()
    app.destroy()


def ShowTanTaiLabelPcdTargetAndPred(points1, colors1, label_point1, label_color1, label_point2, label_color2):
    label_point1[:, 2] -= 5
    label_point2[:, 2] -= 6.5
    pcd1 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(points1))
    pcd1.colors = o3d.pybind.utility.Vector3dVector(colors1)

    # 深绿色：调整绿色通道的值来让颜色变深
    label_color1[:, 0] = 1  # 红色通道
    label_color1[:, 1] = 0  # 绿色通道 (降低值)
    label_color1[:, 2] = 0  # 蓝色通道
    label_pcd = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(label_point1))
    label_pcd.colors = o3d.pybind.utility.Vector3dVector(label_color1)

    # 深蓝色：调整蓝色通道的值来让颜色变深
    label_color2[:, 0] = 0  # 红色通道
    label_color2[:, 1] = 0  # 绿色通道
    label_color2[:, 2] = 1  # 蓝色通道 (降低值)
    label_pcd2 = o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(label_point2))
    label_pcd2.colors = o3d.pybind.utility.Vector3dVector(label_color2)

    mat1 = o3d.visualization.rendering.MaterialRecord()
    mat1.point_size = 5.0  # 设置第一个点云的点大小为5.0
    label_mat = o3d.visualization.rendering.MaterialRecord()
    label_mat.point_size = 12.0  # 设置第二个点云的点大小为10.0

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3d")
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)
    widget3d.scene.add_geometry("pca1", pcd1, mat1)
    widget3d.scene.add_geometry("label_pcd", label_pcd, label_mat)
    widget3d.scene.add_geometry("label_pcd2", label_pcd2, label_mat)
    widget3d.setup_camera(60, widget3d.scene.bounding_box, [0, 0, 0])
    app.run()
    app.destroy()


def ShowTanTaiLabelPcd1(points1, colors1, label_point, label_color):
    # 创建可视化窗口
    mlab.figure(bgcolor=(1, 1, 1))
    pts = mlab.pipeline.scalar_scatter(points1)  # plot the points
    colors11 = np.concatenate((colors1 * 255, (np.ones_like(colors1[:, :1]) * 255)), axis=1).astype(np.uint8)
    pts = mlab.pipeline.scalar_scatter(points1[:, 0], points1[:, 1], points1[:, 2])  # plot the points
    pts.add_attribute(colors11, 'colors')  # assign the colors to each point
    pts.data.point_data.set_active_scalars('colors')
    g = mlab.pipeline.glyph(pts)
    g.glyph.glyph.scale_factor = 1.0  # set scaling for all the points
    g.glyph.scale_mode = 'data_scaling_off'  # make all the points same size

    label_color[:, 0] = 0  # 绿色， 截断点云1，理论上应该完全重合点云1，且总数量少读点云1
    label_color[:, 1] = 0.5
    label_color[:, 2] = 0
    label_colors11 = np.concatenate((label_color * 255, (np.ones_like(label_color[:, :1]) * 255)), axis=1).astype(
        np.uint8)
    label_pts = mlab.pipeline.scalar_scatter(label_point[:, 0], label_point[:, 1], label_point[:, 2])  # plot the points
    label_pts.add_attribute(label_colors11, 'colors')  # assign the colors to each point
    label_pts.data.point_data.set_active_scalars('colors')
    label_g = mlab.pipeline.glyph(label_pts)
    label_g.glyph.glyph.scale_factor = 3  # set scaling for all the points
    label_g.glyph.scale_mode = 'data_scaling_off'  # make all the points same size
    angle = mlab.view(azimuth=0.1)  # 旋转视角

    mlab.show()
