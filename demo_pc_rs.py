import sys
import os
import argparse
import json
from ipdb import set_trace as st
from typing import List
import glob
import torch
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
import trimesh

R_world_to_cam = np.array([
    [ 0, 0, 1],
    [-1, 0, 0],
    [ 0,-1, 0],
], dtype=np.float32)

t_world_to_camera = np.array([0.285, 0., 0.01]) # go2 camera offset in init odom (world) frame

def _read_dataset_scene_graphs(dataset_dir: str, scene_graph_name: str) -> dict:
    """Read central `scene_graph.json` if present, else gather per-step files.

    Returns a dict mapping step_name -> scene_graph (dict).
    """
    entries = {}
    central = os.path.join(dataset_dir, scene_graph_name)
    if os.path.exists(central):
        try:
            with open(central, "r") as f:
                entries = json.load(f)
            return entries
        except Exception as e:
            print(f"Failed to read central scene_graph.json: {e}")

    # fallback: read per-step files
    pattern = os.path.join(dataset_dir, "step_*/scene_graph_*.json")
    for p in glob.glob(pattern):
        try:
            with open(p, "r") as f:
                sg = json.load(f)
            step_name = os.path.basename(os.path.dirname(p))
            entries[step_name] = sg
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
    return entries

def _load_meshes(mesh_paths: list[str]):
    """Combine meshes into a single file at `out_path`.

    Returns True on success.
    """
    if len(mesh_paths) == 0:
        print("No meshes to combine")
        return False

    # Prefer trimesh
    meshes = []
    for p in mesh_paths:
        try:
            m = trimesh.load(p, force="mesh")
            meshes.append(m)
        except Exception as e:
            print(f"trimesh failed to load {p}: {e}")

    return meshes


def scenegraph_to_mesh(scene_graphs: dict):
    """For each step folder, read its scene_graph and write a per-step combined GLB.

    Also writes a final `scene_world_all.glb` that concatenates all per-step meshes.

    if no steps but jsut a big list of objects, just load them all
    """

    if scene_graphs and not any(k.startswith("step_") for k in scene_graphs.keys()):
        # single scene graph case
        all_paths = []
        sg = scene_graphs
        for obj_name, obj in sg.items():
            gw = obj.get("glb_world_path")
            if gw and os.path.exists(gw):
                all_paths.append(gw)

        # load meshes as a list
        if len(all_paths) == 0:
            print("No GLB paths found; nothing to combine")
            return

        meshes = _load_meshes(all_paths)
        return meshes

    else:

        all_paths = []
        # scene_graphs is a dict step->scene_graph
        for step, sg in sorted(scene_graphs.items()):
            if not isinstance(sg, dict):
                continue
            # collect glb paths for this step
            paths = []
            for obj_name, obj in sg.items():
                gw = obj.get("glb_world_path")
                if gw and os.path.exists(gw):
                    paths.append(gw)

            if len(paths) == 0:
                print(f"No GLBs for step {step}, skipping")
                continue

            all_paths.extend(paths)

        # load meshes as a list
        if len(all_paths) == 0:
            print("No GLB paths found across steps; nothing to combine")
            return

        meshes = _load_meshes(all_paths)
        return meshes

class pc_buffer:
    def __init__(self):
        self.buffer = np.empty((0,3))
        self.pcd = o3d.geometry.PointCloud()
        self.voxel_size = 0.01  # 1 cm voxel size for downsampling

    def add_points(self, new_points):
        self.buffer = np.concatenate((self.buffer, new_points), axis=0)
        self.pcd.points = o3d.utility.Vector3dVector(np.array(self.buffer))
        # Downsample the point cloud
        downsampled_pcd = self.pcd.voxel_down_sample(self.voxel_size)
        self.buffer = np.array(downsampled_pcd.points)

    def get_points(self):
        return np.array(self.buffer)

def load_pc_or_depth2pc_and_accumulate(dataset_dir, lidar_dir, scene_graphs, buffer_rs, buffer_lidar):

    odom_init_rs_path = os.path.join(dataset_dir, "step_init", "odom_rs.json")
    with open(odom_init_rs_path, "r") as f:
        init_odom_rs = json.load(f)
    init_odom_rs = np.array([init_odom_rs["t"][0], init_odom_rs["t"][1], 0.0])

    odom_init_lidar_path = os.path.join(dataset_dir, "step_init", "odom_lidar.json")
    with open(odom_init_lidar_path, "r") as f:
        init_odom_lidar = json.load(f)
    init_odom_lidar = np.array([init_odom_lidar["t"][0], init_odom_lidar["t"][1], 0.0])

    # for lidar data
    pc_path = os.path.join(lidar_dir, "accumulated_points.ply")
    if os.path.exists(pc_path):
        print(f"Loading point cloud from {pc_path}")
        pcd = o3d.io.read_point_cloud(pc_path)
        
        # points were accumulated in the lidar init frame so we need to transform to rs init frame
        pcd_points = np.array(pcd.points) + init_odom_lidar - init_odom_rs
        buffer_lidar.add_points(np.array(pcd_points))


    steps_dirs = [d for d in os.listdir(dataset_dir) if d.startswith("step_") and os.path.isdir(os.path.join(dataset_dir, d))]
    for step in sorted(steps_dirs):

        depth_path = os.path.join(dataset_dir, step, "depth.png")
        cam_path = os.path.join(dataset_dir, step, "camera_info.json")
        odom_path = os.path.join(dataset_dir, step, "odom.json")
        if os.path.exists(depth_path) and os.path.exists(cam_path) and os.path.exists(odom_path):
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            with open(cam_path, "r") as f:
                cam_info = json.load(f)
            fx = cam_info["K"][0][0]
            fy = cam_info["K"][1][1]
            cx = cam_info["K"][0][2]
            cy = cam_info["K"][1][2]

            height, width = depth_image.shape
            u = np.arange(width)
            v = np.arange(height)
            uu, vv = np.meshgrid(u, v)

            Z = depth_image.astype(np.float32)
            Z /= 1000.0  # convert from mm to meters
            Z[Z <= 0] = np.nan
            Z[Z > 7.0] = np.nan  # optional: filter out points beyond 10 meters
            X = (uu - cx) * Z / fx
            Y = (vv - cy) * Z / fy

            points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
            points = points[~np.isnan(points).any(axis=1)]

            # transform points to world frame using odom
            with open(odom_path, "r") as f:
                odom = json.load(f)

            points = points @ R_world_to_cam.T  + t_world_to_camera

            t = np.array(odom["t"]) - init_odom_rs
            q = np.array(odom["q"])  # xyzw
            R_world_to_odom = R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            points = points @ R_world_to_odom.T + t

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            buffer_rs.add_points(np.array(pcd.points))
            # yield pcd
        else:
            print(f"No point cloud or depth image found for step {step}")
            continue

    return buffer_rs, buffer_lidar



def accum_pc(dataset_dir, scene_graph_name, lidar_dir):

    buffer_rs = pc_buffer()
    buffer_lidar = pc_buffer()

    scene_graphs = _read_dataset_scene_graphs(dataset_dir, scene_graph_name)
    if len(scene_graphs) == 0:
        print("No scene_graphs found in dataset")
        return

    # load scene graph into list of meshes
    meshes = scenegraph_to_mesh(scene_graphs)
    # load pc
    buffer_rs, buffer_lidar = load_pc_or_depth2pc_and_accumulate(dataset_dir, lidar_dir, scene_graphs, buffer_rs, buffer_lidar)

    # save accumulated pc
    out_pc_path = os.path.join(dataset_dir, "accumulated_points_rs.ply")
    o3d.io.write_point_cloud(out_pc_path, buffer_rs.pcd)
    print(f"Saved accumulated point cloud to {out_pc_path}")

    out_pc_path = os.path.join(dataset_dir, "accumulated_points_lidar.ply")
    o3d.io.write_point_cloud(out_pc_path, buffer_lidar.pcd)
    print(f"Saved accumulated point cloud to {out_pc_path}")

    # saved meshes and pc together
    all_rs_meshes = meshes.copy()
    pc_mesh = trimesh.PointCloud(buffer_rs.get_points())
    all_rs_meshes.append(pc_mesh)
    scene = trimesh.Scene(all_rs_meshes)
    out_combined_path = os.path.join(dataset_dir, "combined_mesh_and_rs_pc.glb")
    # Exporting the Scene directly
    with open(out_combined_path, 'wb') as f:
        f.write(scene.export(file_type='glb'))
    print(f"Saved combined mesh and point cloud to {out_combined_path}")

    all_lidar_meshes = meshes.copy()
    pc_mesh = trimesh.PointCloud(buffer_lidar.get_points())
    all_lidar_meshes.append(pc_mesh)
    scene = trimesh.Scene(all_lidar_meshes)
    out_combined_path = os.path.join(dataset_dir, "combined_mesh_and_lidar_pc.glb")
    # Exporting the Scene directly
    with open(out_combined_path, 'wb') as f:
        f.write(scene.export(file_type='glb'))
    print(f"Saved combined mesh and point cloud to {out_combined_path}")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to dataset")
    parser.add_argument("lidar_dir", type=str, help="Path to lidar directory")
    parser.add_argument("--scene_graph_name", type=str, default="scene_graph.json", help="Path to scene graph JSON file name")
    args = parser.parse_args()

    """
    python3 demo_go2.py <dataset_dir> [--max_steps N] [--pointmap_mode]
    """

    dataset_dir = args.dataset_dir
    scene_graph_name = args.scene_graph_name
    lidar_dir = args.lidar_dir
    accum_pc(dataset_dir, scene_graph_name, lidar_dir)