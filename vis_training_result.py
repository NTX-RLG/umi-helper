import sys
import os
import time
import argparse
import pickle
import numpy as np
import torch
import dill
import hydra
import cv2

import open3d as o3d

from torch.utils.data import DataLoader
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_obs_resolution, get_real_umi_action
from diffusion_policy.common.pytorch_util import dict_apply
from umi.common.pose_util import rot6d_to_mat, pose_to_mat
import omegaconf


class PolicyInference:
    def __init__(self, ckpt_path: str, output_dir: str, device: str):
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        if not self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.join(self.ckpt_path, 'checkpoints', 'latest.ckpt')
        payload = torch.load(open(self.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']
        # export cfg to yaml
        cfg_path = self.ckpt_path.replace('.ckpt', '.yaml')
        with open(cfg_path, 'w') as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
            print(f"Exported config to {cfg_path}")
        print(f"Loading configure: {self.cfg.name}, workspace: {self.cfg._target_}, policy: {self.cfg.policy._target_}, model_name: {self.cfg.policy.obs_encoder.model_name}")
        self.obs_res = get_real_obs_resolution(self.cfg.task.shape_meta)

        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace = cls(self.cfg)
        self.workspace: BaseWorkspace
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy:BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
            print("Using EMA model")
        self.policy.num_inference_steps = 16
        
        obs_pose_rep = self.cfg.task.pose_repr.obs_pose_repr
        action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        print('obs_pose_rep', obs_pose_rep)
        print('action_pose_repr', action_pose_repr)
        
        self.device = torch.device(device)

    def set_dataloader(self, dataset_path: str):
        cfg = self.cfg
        cfg.task.dataset['dataset_path'] = dataset_path
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        cfg.dataloader.batch_size = 1
        # cfg.dataloader.shuffle = False
        self.dataloader = DataLoader(dataset, **cfg.dataloader)
        # normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        # normalizer = dataset.get_normalizer()
        # self.policy.set_normalizer(normalizer)
        self.policy.eval().to(self.device)
        self.policy.reset()

    def predict_action(self, obj_dict):
        with torch.no_grad():
            action = self.policy.predict_action(obj_dict)
        return action


def main(input_dir, output, device, replay_buffer=None, video=None):
    node = PolicyInference(input_dir, output, device)
    node.set_dataloader(dataset_path=replay_buffer)
    num = 0
    vis = o3d.visualization.Visualizer()
    for batch_idx, batch in enumerate(node.dataloader):
        if batch_idx < num:
            continue
        num += 30
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        gt_action = batch['action']
        pred_action = node.policy.predict_action(batch['obs'])['action_pred']
        saved_folder = os.path.join(node.output_dir, "vis_on_trained_image", f"{batch_idx}")
        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
        image0 = batch['obs']['camera0_rgb'][0, 0].cpu()
        image0 = (image0.permute(1, 2, 0) * 255).byte().numpy()
        image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(saved_folder, 'image0.jpg'), image0)
        image1 = batch['obs']['camera0_rgb'][0, 1].cpu()
        image1 = (image1.permute(1, 2, 0) * 255).byte().numpy()
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(saved_folder, 'image1.jpg'), image1)
        cv2.imshow('image', image0)
        cv2.waitKey(0)
        # T = 16，预测16步，每一步有10个数据,3个pos，6个rot，1个width
        pos = pred_action[..., :3].cpu().detach().numpy()[0]  # [16, 3]
        rot = rot6d_to_mat(pred_action[..., 3:9].cpu().detach().numpy())[0]  # [16, 3, 3]
        width = pred_action[..., 9].cpu().detach().numpy()[0]  # [16,]
        # Create a coordinate frame with pos as the origin and rot as the rotation matrix
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

        pos = gt_action[..., :3].cpu().detach().numpy()[0]  # [16, 3]
        rot = rot6d_to_mat(gt_action[..., 3:9].cpu().detach().numpy())[0]  # [16, 3, 3]
        width = gt_action[..., 9].cpu().detach().numpy()[0]  # [16,]
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
            line_set.translate([0.12, 0.0, 0.0])
            colors = [[0, 1 - 0.07 * i, 0] for _ in range(len(lines))]
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
        line_set.translate([0.12, 0.0, 0.0])
        
        # 设置线的颜色
        colors = [[1, 0, 0] for _ in range(len(lines))]  # 红色
        line_set.colors = o3d.utility.Vector3dVector(colors)
        coord_frames.append(line_set)

        file_path = os.path.join(saved_folder, 'gt_est.txt')
        print(f"{saved_folder}")
        with open(file_path, 'w') as file:
            file.write(f"gt_action:\n{gt_action}\n")
            file.write(f"pred_action:\n{pred_action}\n")
        vis.create_window()
        vis.clear_geometries()
        for frame in coord_frames:
            vis.add_geometry(frame)
        ctr = vis.get_view_control()
        ctr.set_front([-0.2, -0.2, -1])
        ctr.set_lookat([0.06, 0, 0])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.35)
        vis.run()
        vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to checkpoint')
    parser.add_argument('--output', '-o', required=True, help='RefPath to output directory')
    parser.add_argument('--device', default="cuda", help="Device to run on")
    parser.add_argument('--replay_buffer', default=None, help="RefPath to replay buffer")
    parser.add_argument('--video', default=None, help="RefPath to video")
    args = parser.parse_args()
    assert args.replay_buffer is not None or args.video is not None
    assert not (args.replay_buffer is not None and args.video is not None), "only"
    main(args.input, args.output, args.device, replay_buffer=args.replay_buffer, video=args.video)


