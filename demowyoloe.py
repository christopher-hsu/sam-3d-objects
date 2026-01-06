import sys, os
from ultralytics import YOLO
from ipdb import set_trace as st
from ultralytics.utils.checks import collect_system_info

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import cv2

# https://github.com/facebookresearch/sam-3d-objects/blob/eb83f583573bff596fdfdfe98c1e883335b7aa29/notebook/demo_aligned_pointmap.ipynb

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, display_image, make_scene, ready_gaussian_for_video_rendering

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# yoloe
model = YOLO("yoloe-11l-seg-pf.pt")

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
size = (640, 480)
image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

results = model.predict(image, half=True, imgsz=(480,640))
result = results[0] # single image input
result.save()

all_labels = result.names

scene_graph = {}
labels = []
for box in result.boxes:
    label_id = int(box.cls)
    conf = box.conf.item()
    samantic_label = all_labels[label_id]
    samantic_label = samantic_label.replace(" ", "_") # replace spaces with underscores
    labels.append(samantic_label)

    if samantic_label not in scene_graph:
        scene_graph[samantic_label] = {}
        scene_graph[samantic_label]["confidence"] = conf


masks = results[0].masks.data.cpu().numpy().astype(bool)

# run model
outputs = [inference(image, mask, seed=42) for mask in masks]

# # export gaussian splat
for i, output in enumerate(outputs):
    output["gs"].save_ply(f"notebook/gaussians/{labels[i]}.ply")
    scene_graph[labels[i]]["ply_path"] = f"notebook/gaussians/{labels[i]}.ply"


scene_gs = make_scene(*outputs)
scene_gs = ready_gaussian_for_video_rendering(scene_gs,fix_alignment=False)
scene_gs.save_ply("notebook/gaussians/multi/scene_splat.ply")