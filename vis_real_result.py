import os
import argparse
import numpy as np
import cv2
import glob
import open3d as o3d

from umi.common.pose_util import rot6d_to_mat


def main(video=None):
    assert os.path.exists(video)
    iter_list = [int(p.split("/")[-1].split(".")[0]) for p in glob.glob(os.path.join(video, "*.npy"))]
    iter_list.sort()
    vis = o3d.visualization.Visualizer()
    for iter in iter_list:
        img_path = os.path.join(video, f"{iter}.jpg")
        action_path = os.path.join(video, f"{iter}.npy")
        img = cv2.imread(img_path)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        action = np.load(action_path, allow_pickle=True).item()
        raw_action = action['raw_action']  # [16, 10]
        pos = raw_action[..., :3]  # [16, 3]
        rot = rot6d_to_mat(raw_action[..., 3: 9])  # [16, 3, 3]
        width = raw_action[..., 9]  # [16,]
        coord_frames = []
        for i in range(len(pos)):
            w = width[i] / 2
            vertices = np.array([
                [w, 0, 0],  # 顶点 1
                [-w, 0, 0],  # 顶点 2
                [w, 0, 0.1],  # 顶点 3
                [-w, 0, 0.1],   # 顶点 4
                [0, 0, 0],
                [0, 0.01, 0]
            ])

            # 定义矩形框的边（线段）
            lines = np.array([
                [0, 1],  # 边 1
                [0, 2],  # 边 2
                [1, 3],  # 边 3
                [4, 5]   # 边 4
            ])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.rotate(rot[i])
            line_set.translate(pos[i])
            colors = [[0, 0, 1 - 0.07 * i] for _ in range(len(lines))]  # 红色
            line_set.colors = o3d.utility.Vector3dVector(colors)
            coord_frames.append(line_set)

        # 定义矩形框的顶点
        vertices = np.array([
            [0.04, 0.02, 0],  # 顶点 1
            [0.04, -0.02, 0],  # 顶点 2
            [-0.04, -0.02, 0],  # 顶点 3
            [-0.04, 0.02, 0]   # 顶点 4
        ])

        # 定义矩形框的边（线段）
        lines = np.array([
            [0, 1],  # 边 1
            [1, 2],  # 边 2
            [2, 3],  # 边 3
            [3, 0]   # 边 4
        ])

        # 创建 LineSet 对象
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # 设置线的颜色
        colors = [[1, 0, 0] for _ in range(len(lines))]  # 红色
        line_set.colors = o3d.utility.Vector3dVector(colors)
        coord_frames.append(line_set)
        vis.create_window()
        vis.clear_geometries()
        for frame in coord_frames:
            vis.add_geometry(frame)
        ctr = vis.get_view_control()
        ctr.set_front([-0.2, -0.2, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.35)
        vis.run()
        vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None, help="RefPath to video")
    args = parser.parse_args()
    main(video=args.video)


