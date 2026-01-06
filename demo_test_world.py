import sys
import os
import argparse
import json
from ultralytics import YOLO
from ipdb import set_trace as st
import torch
import cv2
from pytorch3d.transforms import quaternion_to_matrix, Transform3d
from scipy.spatial.transform import Rotation as R


import open3d as o3d
import numpy as np
import trimesh

from demo_scenegraph import combine_per_step



# Z-up → Y-up conversion
R_zup_to_yup = torch.tensor([
    [-1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
], dtype=torch.float32)

R_yup_to_zup = R_zup_to_yup.T

# flip Z-axis 
R_flip_z = torch.tensor([
    [1, 0, 0],
    [ 0, 1, 0],
    [ 0, 0, -1],
], dtype=torch.float32)

# Convert from pointmap convention [-X, -Y, Z] back to true
R_pytorch3d_to_cam = torch.tensor([
    [-1,  0,  0],  
    [ 0, -1,  0],  
    [ 0,  0,  1], 
], dtype=torch.float32)

"""
# Rotation from world (x forward, y left, z up) to camera frame (x right , y down, z forward)
static_cam_offset_euler = np.radians([-90, 0, 90]) # Example: X-90, Y-0, Z+90 (XYZ order)
R_static_world_to_cam = R.from_euler('xyz', static_cam_offset_euler, degrees=False).as_matrix()
# additional rotation to make the camera frame point forward
R_additional_rotation = R.from_euler('xyz', np.radians([0, 0, 180]), degrees=False).as_matrix()
R_world_to_cam = R_additional_rotation @ R_static_world_to_cam 
"""
R_world_to_cam = np.array([
    [ 0, 0, 1],
    [-1, 0, 0],
    [ 0,-1, 0],
], dtype=np.float32)

t_world_to_camera = np.array([0.285, 0., 0.01]) # go2 camera offset in init odom (world) frame

def transform_mesh_vertices(vertices, rotation, translation, scale):
    """
    Transform mesh vertices from local object space to world/camera frame:

    1. Flip Z-axis ( depending on GLB orientation)
    2. Convert from Y-up (GLB) to Z-up (canonical PyTorch3D frame)
    3. Apply GS outputs: scale, rotation, translation
    4. Convert back to Y-up for GLB export
    """

    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)

    vertices = vertices.unsqueeze(0)  #  batch dimension [1, N, 3]

    # Flip Z-axis
    vertices = vertices @ R_flip_z.to(vertices.device) 

    # Convert mesh from Y-up (GLB) → Z-up (canonical PyTorch3D)
    vertices = vertices @ R_yup_to_zup.to(vertices.device)

    # apply gaussian splatting transformations 
    R_mat = quaternion_to_matrix(rotation.to(vertices.device))
    tfm = Transform3d(dtype=vertices.dtype, device=vertices.device)
    tfm = (
        tfm.scale(scale)
           .rotate(R_mat)
           .translate(translation[0], translation[1], translation[2])
    )
    vertices_world = tfm.transform_points(vertices)

    # convert back to Y-up so GLB is saved correctly
    vertices = vertices @ R_zup_to_yup.to(vertices.device)

    # remove batch dimension
    return vertices_world[0]  

def transform_glb(out):
    mesh = out["glb"]
    vertices = mesh.vertices

    vertices_tensor = torch.tensor(vertices)

    S = out["scale"][0].cpu().float()
    T = out["translation"][0].cpu().float()
    R = out["rotation"].squeeze().cpu().float()

    # Transform vertices
    vertices_transformed = transform_mesh_vertices(vertices, R, T, S)
  
    # --- Convert vertices from pointmap frame back to true camera frame ---
    # (undoing the earlier pointmap conversion: [-X, -Y, Z])
    vertices_transformed = vertices_transformed @ R_pytorch3d_to_cam.to(vertices_transformed.device)

    # Update mesh vertices
    mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)

    return mesh


def transform_glb_to_world(mesh, odom, init_odom=None):
    """Transform a mesh (vertices in camera frame (x right, y down, z forward)) 
    into world frame (x forward, y left, z up) into odom (body) frame.

    `odom` is expected to be a dict with keys `t` (list of 3) and `q` (list of 4: x,y,z,w).
    Returns a new mesh object with transformed vertices.
    """

    if odom is None:
        return mesh

    t = np.array(odom.get("t"))
    # origin offset. make step_0001 to be at (0,0,z)
    t_init = np.array([init_odom.get("t")[0], init_odom.get("t")[1], 0]) if init_odom is not None else np.array([0, 0, 0])
    t -= t_init

    q = odom.get("q")
    if t is None or q is None:
        return mesh

    verts = np.asarray(mesh.vertices).astype(np.float32)

    # camera frame to world frame
    verts_world = verts @ R_world_to_cam.T + t_world_to_camera
    
    # world to odom frame
    R_world_to_odom = R.from_quat(q).as_matrix()
    verts_odom = verts_world @ R_world_to_odom.T + t

    mesh_copy = mesh.copy()
    mesh_copy.vertices = verts_odom
    return mesh_copy

def depth_to_pointmap(depth_image, cam_info):
    """Convert depth image to pointmap (HxWx3) using camera intrinsics.
    https://github.com/facebookresearch/sam-3d-objects/blob/eb83f583573bff596fdfdfe98c1e883335b7aa29/notebook/demo_aligned_pointmap.ipynb
    """

    fx = cam_info["K"][0][0]
    fy = cam_info["K"][1][1]
    cx = cam_info["K"][0][2]
    cy = cam_info["K"][1][2]

    height, width = depth_image.shape
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)

    Z = depth_image.astype(np.float32)
    Z[Z <= 0] = np.nan
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    # ---------------------------------------------------------------------
    # Convert image coordinates (x→right, y→down) into PyTorch3D coordinates:
    #   PyTorch3D expects a right-handed camera frame with:
    #       +x → right, +y → UP, +z → forward.
    #   So we flip both X and Y:
    #       -Y  converts image Y-down into Y-up,
    #       -X  keeps the coordinate system right-handed.
    # ---------------------------------------------------------------------
    pointmap = np.stack((-X, -Y, Z), axis=-1)  # HxWx3
    pointmap = torch.tensor(pointmap, dtype=torch.float32)
    return pointmap

def load_step(step_path):
    """Load rgb image, odom json, and point cloud ply (or points.npy) from a step folder."""
    data = {"rgb": None, "odom": None, "pc": None, "pointmap": None}

    # resize image and pointmap
    size = (640, 480)

    rgb_path = os.path.join(step_path, "rgb.png")
    if os.path.exists(rgb_path):
        img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        # convert BGR->RGB
        if img is not None and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        data["rgb"] = img

    odom_path = os.path.join(step_path, "odom.json")
    if os.path.exists(odom_path):
        with open(odom_path, "r") as f:
            data["odom"] = json.load(f)

    ply_path = os.path.join(step_path, "points.ply")
    npy_path = os.path.join(step_path, "points.npy")
    if os.path.exists(ply_path) and o3d is not None:
        pc = o3d.io.read_point_cloud(ply_path)
        data["pc"] = pc
    elif os.path.exists(npy_path):
        pts = np.load(npy_path)
        data["pc"] = pts

    camera_info_path = os.path.join(step_path, "camera_info.json")
    if os.path.exists(camera_info_path):
        with open(camera_info_path, "r") as f:
            camera_info = json.load(f)
    
    depth_path = os.path.join(step_path, "depth.png")
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        pointmap = depth_to_pointmap(depth, camera_info)
        data["pointmap"] = pointmap

    return data


def process_dataset(dataset_dir, max_steps=None, pointmap_mode=True):
    # gather step folders
    steps = [d for d in os.listdir(dataset_dir) if d.startswith("step_")]
    steps.sort()

    if max_steps is not None:
        steps = steps[:max_steps]

    out_gauss_dir = os.path.join(dataset_dir, "gaussians")
    out_glb_dir = os.path.join(dataset_dir, "glbs")
    os.makedirs(out_glb_dir, exist_ok=True)
    os.makedirs(out_gauss_dir, exist_ok=True)
    # central scene graph for entire dataset
    dataset_scene_graph = {}

    for i, s in enumerate(steps):
        step_path = os.path.join(dataset_dir, s)
        print(f"Processing {s} ({i+1}/{len(steps)})")
        data = load_step(step_path)
        img = data["rgb"]
        odom = data["odom"]
        pc = data["pc"]
        pointmap = data["pointmap"]

        if i == 0:
            init_odom = odom

        if img is None:
            print(f"  No RGB image in {s}, skipping")
            continue

        # load scene graph for this step if exists
        scene_graph_path = os.path.join(step_path, f"scene_graph_{s}.json")
        if os.path.exists(scene_graph_path):
            with open(scene_graph_path, "r") as f:
                scene_graph = json.load(f)
        else:
            scene_graph = {}

        # process each object in scene graph by importing the glb from glb_path
        for samantic_label, obj in scene_graph.items():
            glb_path = obj.get("glb_path")

            if glb_path is None or not os.path.exists(glb_path):
                print(f"  No glb_path for {samantic_label}, skipping")
                continue

            glb_mesh = trimesh.load(glb_path, force='mesh')

            # transform into world using odom and save into per-step glb_world folder
            glb_world_step_dir = os.path.join(dataset_dir, "glbs_world", s)
            os.makedirs(glb_world_step_dir, exist_ok=True)
            if odom is not None:
                try:
                    glb_world_mesh = transform_glb_to_world(glb_mesh, odom, init_odom)
                    glb_world_out = os.path.join(glb_world_step_dir, f"{samantic_label}.glb")
                    glb_world_mesh.export(glb_world_out)
                    scene_graph[samantic_label]["glb_world_path"] = glb_world_out
                except Exception as e:
                    print(f"  Failed to export world glb for {samantic_label}: {e}")


        step_glb_dir = os.path.join(out_glb_dir, s)
        cam_glb_paths = [os.path.join(step_glb_dir, f) for f in os.listdir(step_glb_dir) if f.endswith('.glb')]
        cam_glb_paths = [p for p in cam_glb_paths if os.path.exists(p)]
        cam_combined_out = os.path.join(out_glb_dir, f"scene_{s}.glb")
        if len(cam_glb_paths) > 0:
            meshes = []
            for p in cam_glb_paths:
                try:
                    m = trimesh.load(p, force='mesh')
                    meshes.append(m)
                except Exception as e:
                    print(f"  trimesh failed to load {p}: {e}")
            if len(meshes) > 0:
                combined = trimesh.util.concatenate(meshes)
                combined.export(cam_glb_paths and cam_combined_out)
                print(f"  Saved camera-frame combined GLB to {cam_combined_out}")


        # save scene_graph for this step (per-step)
        scene_graph_path = os.path.join(step_path, f"scene_graph_{s}.json")
        with open(scene_graph_path, "w") as f:
            json.dump(scene_graph, f, indent=2)


        # update central dataset-level scene graph
        dataset_scene_graph[s] = scene_graph
        dataset_scene_graph_path = os.path.join(dataset_dir, "scene_graph.json")
        with open(dataset_scene_graph_path, "w") as f:
            json.dump(dataset_scene_graph, f, indent=2)

        # create combined GLB of world glbs for this step (if trimesh available)
        glb_world_dir = os.path.join(dataset_dir, "glbs_world")
        combined_world_out = os.path.join(glb_world_dir, f"scene_world_{s}.glb")
        # collect world glbs listed in scene_graph
        world_paths = [v.get("glb_world_path") for v in scene_graph.values() if v.get("glb_world_path")]
        meshes = []
        for p in world_paths:
            m = trimesh.load(p, force="mesh")
            meshes.append(m)

        if len(meshes) > 0:
            combined = trimesh.util.concatenate(meshes)
            combined.export(combined_world_out)
            print(f"  Saved combined world GLB to {combined_world_out}")


        # print odom info summary
        if odom is not None:
            # print(f"  odom t: {odom.get('t')} q: {odom.get('q')}")
            t = odom.get("t")
            t_init = np.array([init_odom.get("t")[0], init_odom.get("t")[1], 0]) if init_odom is not None else np.array([0, 0, 0])
            t -= t_init
            print(f"  odom t (offset): {t} q: {odom.get('q')}")


        # if pointcloud is loaded, print a summary
        if pc is not None:
            if o3d is not None and isinstance(pc, o3d.geometry.PointCloud):
                print(f"  pointcloud has {len(pc.points)} points (ply)")
            else:
                pts = np.asarray(pc)
                print(f"  pointcloud loaded with shape {pts.shape}")

    # After processing all steps, attempt to build per-step and dataset-level combined GLBs
    if combine_per_step is not None:
        try:
            print("Running combine_per_step from demo_scenegraph to create combined GLBs...")
            combine_per_step(dataset_dir)
        except Exception as e:
            print(f"combine_per_step failed: {e}")
    else:
        print("demo_scenegraph.combine_per_step not available; skipping final combine step")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to dataset")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to process")
    parser.add_argument("--pointmap_mode", type=bool, default=False, help="Enable pointmap mode")
    args = parser.parse_args()

    """
    python3 demo_go2.py <dataset_dir> [--max_steps N] [--pointmap_mode]
    """

    dataset_dir = args.dataset_dir
    max_steps = args.max_steps
    pointmap_mode = args.pointmap_mode
    process_dataset(dataset_dir, max_steps=max_steps, pointmap_mode=pointmap_mode)