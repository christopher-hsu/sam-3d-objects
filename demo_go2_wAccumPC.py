import sys
import os
import argparse
import json
from ultralytics import YOLO
from ipdb import set_trace as st

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import cv2
from pytorch3d.transforms import quaternion_to_matrix, Transform3d
from scipy.spatial.transform import Rotation as R

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, display_image, make_scene, ready_gaussian_for_video_rendering

import open3d as o3d
import numpy as np
import trimesh


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

def lidar2depth(lidar_pts, cam_info, odom_info, init_odom):
    """
    convert lidar points in world into the camera frame
    # """
    # img_height, img_width = (
    #     cam_info["height"],
    #     cam_info["width"],
    # )  # Replace with your camera resolution
    img_height = 480
    img_width = 640

    depth_image = np.zeros((img_height, img_width), dtype=np.uint16)

    # --- Transform lidar points from world to camera frame ---
    # world to odom
    t_odom = np.array(odom_info.get("t")) - np.array([init_odom.get("t")[0], init_odom.get("t")[1], 0])
    q_odom = odom_info.get("q")
    R_world_to_odom = R.from_quat(q_odom).as_matrix()

    lidar_odom = (lidar_pts - t_odom) @ R_world_to_odom
    # odom to camera
    lidar_cam = (lidar_odom - t_world_to_camera) @ R_world_to_cam

    # Filter out points behind the camera (Z > 0)
    lidar_cam = lidar_cam[lidar_cam[:, 2] > 0]
    if lidar_cam.shape[0] == 0:
        return depth_image, None, None # No points in front of camera

    """Intrinsics"""
    K = np.array(cam_info["K"])  # 3x3 camera intrinsic matrix
    # scale instrinsics if image size is different
    K[0, 0] *= img_width / cam_info["width"]
    K[1, 1] *= img_height / cam_info["height"]
    K[0, 2] *= img_width / cam_info["width"]
    K[1, 2] *= img_height / cam_info["height"]
    # project to image plane
    uvw = K @ lidar_cam.T
    # normalize by w
    uv = uvw[:2] / uvw[2]

    # where Z is the distance from the image plane and scaling by 256.0 for int16
    uvz = np.vstack((uv, lidar_cam[:, 2] * 256.0)).T
    # filter out points outside the image
    uvz_in = uvz[
        (0 <= uvz[:, 0])
        & (uvz[:, 0] < img_width)
        & (0 <= uvz[:, 1])
        & (uvz[:, 1] < img_height)
    ]
    # put valid points in image
    depth_image[uvz_in[:, 1].astype(int), uvz_in[:, 0].astype(int)] = (uvz_in[:, 2]*1000.).astype(np.uint16)

    return depth_image.astype(np.uint16)

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

    if mesh.volume < 0: # fixes surface normals
        mesh.invert()

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

    if mesh_copy.volume < 0: # fixes surface normals
        mesh_copy.invert()

    obj_info = {}
    obj_info['centroid'] = mesh_copy.centroid.tolist()
    obj_info['aabb_bounds'] = mesh_copy.bounds.tolist()
    # obj_info['extents'] = mesh_copy.extents.tolist()
    return mesh_copy, obj_info

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
    Z /= 1000.0  # convert from mm to meters
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

def load_step(step_path, pc_path=None, index=0):
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

    if index == 0:
        # save init_odom as global variable
        global init_odom
        init_odom = data["odom"]

    # ply_path = os.path.join(step_path, "points.ply")
    # npy_path = os.path.join(step_path, "points.npy")
    if os.path.exists(pc_path):
        pc = o3d.io.read_point_cloud(pc_path)
        data["pc"] = pc

    camera_info_path = os.path.join(step_path, "camera_info.json")
    if os.path.exists(camera_info_path):
        with open(camera_info_path, "r") as f:
            camera_info = json.load(f)
    
    #accum points has already been shifted by init_odom cuz not clean code
    depth_img = lidar2depth(np.asarray(pc.points), camera_info, data["odom"], init_odom)

    depth_path = os.path.join(step_path, "depth_pc.png")
    #save depth image for visualization
    cv2.imwrite(depth_path, depth_img)
    # if os.path.exists(depth_path):
        # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    pointmap = depth_to_pointmap(depth_img, camera_info)
    data["pointmap"] = pointmap

    return data

def _combine_meshes(glb_paths, out_path):
    if len(glb_paths) > 0:
        meshes = []
        for p in glb_paths:
            m = trimesh.load(p, force='mesh')
            meshes.append(m)
        if len(meshes) > 0:
            combined = trimesh.util.concatenate(meshes)
            combined.export(out_path)
            print(f"  Saved camera-frame combined GLB to {out_path}")

def process_dataset(dataset_dir, max_steps=None, pointmap_mode=True, min_conf=0.5):
    # load models
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # yoloe model (segmentation)
    model = YOLO("yoloe-11l-seg-pf.pt")

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

    pc_path = os.path.join(dataset_dir, "accumulated_points.ply")

    all_paths = []  # for final combined GLB
    for i, s in enumerate(steps):
        step_path = os.path.join(dataset_dir, s)
        print(f"Processing {s} ({i+1}/{len(steps)})")
        data = load_step(step_path, pc_path, i)
        img = data["rgb"]
        odom = data["odom"]
        pc = data["pc"]
        pointmap = data["pointmap"]

        if i == 0:
            init_odom = odom

        if img is None:
            print(f"  No RGB image in {s}, skipping")
            continue

        # run yolo segmentation
        # model.predict expects HxWxC numpy (RGB)
        results = model.predict(img, half=True, imgsz=(480, 640))
        result = results[0]
        result.save(filename=f"{step_path}/rgb_pf.png")

        # get names and masks
        all_labels = result.names

        masks = results[0].masks.data.cpu().numpy().astype(bool)

        labels = []
        scene_graph = {}

        if masks is None or len(masks) == 0:
            print(f"  No segmentation masks for {s}")
            continue

        outputs = []
        for mi, mask in enumerate(masks):
            # get label
            try:
                box = result.boxes[mi]
                label_id = int(box.cls)
                conf = box.conf.item()
                samantic_label = all_labels[label_id].replace(" ", "_")
            except Exception:
                continue

            # only include obj if confidence > min_conf
            if conf > min_conf:
                if samantic_label not in scene_graph:
                    scene_graph[samantic_label] = {"confidence": conf}
                else:
                    samantic_label = f"{samantic_label}_{mi}"
                    scene_graph[samantic_label] = {"confidence": conf}
            else:
                print(f"  Skipping {samantic_label} with low confidence {conf:.2f} < {min_conf}")
                continue

            labels.append(samantic_label)

            # call inference pipeline
            if pointmap_mode == True:
                out = inference(img, mask, seed=42, pointmap=pointmap)
            else:
                out = inference(img, mask, seed=42)
            outputs.append(out)
            """
            out:
            {
                "6drotation_normalized"
                "scale"
                "shape"
                "translation"
                "traslation_scale"
                "coords_original"
                "coords"
                "rotation"
                "mesh"
                "gaussian"
                "glb": trimesh object,
                "gs"
                "pointmap"
                "pointmap_colors"
            }
            """
            
            # ensure per-step output directories
            step_gauss_dir = os.path.join(out_gauss_dir, s)
            step_glb_dir = os.path.join(out_glb_dir, s)
            os.makedirs(step_gauss_dir, exist_ok=True)
            os.makedirs(step_glb_dir, exist_ok=True)

            # save ply for this object into per-step gaussian folder
            ply_out = os.path.join(step_gauss_dir, f"{samantic_label}.ply")
            out["gs"].save_ply(ply_out)
            scene_graph[samantic_label]["ply_path"] = ply_out

            # save glb for this object into per-step glb folder
            glb_mesh = transform_glb(out)
            glb_out = os.path.join(step_glb_dir, f"{samantic_label}.glb")
            glb_mesh.export(glb_out)
            scene_graph[samantic_label]["glb_path"] = glb_out

            # transform into world using odom and save into per-step glb_world folder
            glb_world_step_dir = os.path.join(dataset_dir, "glbs_world", s)
            os.makedirs(glb_world_step_dir, exist_ok=True)
            if odom is not None:
                glb_world_mesh, obj_info = transform_glb_to_world(glb_mesh, odom, init_odom)
                glb_world_out = os.path.join(glb_world_step_dir, f"{samantic_label}.glb")
                glb_world_mesh.export(glb_world_out)
                scene_graph[samantic_label]["glb_world_path"] = glb_world_out
                scene_graph[samantic_label]["obj_in_world_info"] = obj_info

        # create combined gs scene and save
        if len(outputs) > 0:
            scene_gs = make_scene(*outputs)
            scene_gs = ready_gaussian_for_video_rendering(scene_gs, fix_alignment=False)
            multi_out = os.path.join(out_gauss_dir, f"scene_{s}.ply")
            scene_gs.save_ply(multi_out)
            print(f"  Saved scene ply to {multi_out}")

        # Camera-frame combined GLB (under `glbs/scene_{s}.glb`)
        step_glb_dir = os.path.join(out_glb_dir, s)
        try:
            cam_glb_paths = [os.path.join(step_glb_dir, f) for f in os.listdir(step_glb_dir) if f.endswith('.glb')]
        except:
            print("  No GLB files probably due to all confidence below threshold")
            continue
        cam_glb_paths = [p for p in cam_glb_paths if os.path.exists(p)]
        cam_combined_out = os.path.join(out_glb_dir, f"scene_{s}.glb")
        _combine_meshes(cam_glb_paths, cam_combined_out)

        # World-frame combined GLB for this step (under `glbs_world/scene_world_{s}.glb`)
        glb_world_dir = os.path.join(dataset_dir, "glbs_world")
        combined_world_step_out = os.path.join(glb_world_dir, f"scene_world_{s}.glb")
        # collect world glbs listed in scene_graph
        world_paths = [v.get("glb_world_path") for v in scene_graph.values() if v.get("glb_world_path")]
        _combine_meshes(world_paths, combined_world_step_out)

        # collect all paths for final dataset-level combined GLB
        all_paths.extend(world_paths)

        # save scene_graph for this step (per-step)
        scene_graph_path = os.path.join(step_path, f"scene_graph_{s}.json")
        with open(scene_graph_path, "w") as f:
            json.dump(scene_graph, f, indent=2)

        # update central dataset-level scene graph
        dataset_scene_graph[s] = scene_graph
        dataset_scene_graph_path = os.path.join(dataset_dir, "scene_graph.json")
        with open(dataset_scene_graph_path, "w") as f:
            json.dump(dataset_scene_graph, f, indent=2)

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

    # After processing all steps, build dataset-level combined GLBs
    combined_all_steps_out = os.path.join(glb_world_dir, "scene_world_all.glb")
    _combine_meshes(all_paths, combined_all_steps_out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to dataset")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps to process")
    parser.add_argument("--pointmap_mode", type=bool, default=False, help="Enable pointmap mode")
    parser.add_argument("--min_conf", type=float, default=0.5, help="Minimum confidence for segmentation")
    args = parser.parse_args()

    """
    python3 demo_go2.py <dataset_dir> [--max_steps N] [--pointmap_mode]
    """

    dataset_dir = args.dataset_dir
    max_steps = args.max_steps
    pointmap_mode = args.pointmap_mode
    process_dataset(dataset_dir, max_steps=max_steps, pointmap_mode=pointmap_mode)