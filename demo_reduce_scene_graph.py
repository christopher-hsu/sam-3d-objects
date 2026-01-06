import json
import numpy as np
import uuid
import os
import torch
import re # Import the regular expression library
from transformers import CLIPModel, CLIPTokenizer
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- CLIP Model Setup ---
# *** IMPORTANT: This setup requires 'torch' and 'transformers' libraries. ***
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

CLIP_MODEL = None
CLIP_TOKENIZER = None

def initialize_clip_model():
    """Initializes the global CLIP model and tokenizer."""
    global CLIP_MODEL, CLIP_TOKENIZER
    print(f"Loading CLIP model '{MODEL_NAME}' on device: {DEVICE}...")
    try:
        CLIP_MODEL = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading CLIP model. Please ensure you have 'torch' and 'transformers' installed.")
        print(f"Error: {e}")
        # Fallback to the simple string match if CLIP fails to load
        CLIP_MODEL = False
        CLIP_TOKENIZER = False


def get_text_embedding(text: str) -> torch.Tensor:
    """Computes the CLIP text embedding for a given string."""
    if not CLIP_MODEL:
        raise RuntimeError("CLIP Model not initialized.")

    cleaned_text = clean_label(text)
    inputs = CLIP_TOKENIZER([cleaned_text], padding=True, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # Get the text features (embeddings)
        text_features = CLIP_MODEL.get_text_features(**inputs)
    
    # Normalize the features to unit length (standard for cosine similarity)
    return text_features / text_features.norm(p=2, dim=-1, keepdim=True)


def clean_label(label: str) -> str:
    """
    Cleans labels by converting to lowercase and removing common suffixes 
    like '_1', '_2', '_A', '_B', or simple sequential numbers.
    
    Example: 'chair_1' -> 'chair'
             'table_A' -> 'table'
             'Couch 1' -> 'couch'
    """
    label = label.lower().strip()
    
    # Regex 1: Remove '_number' or ' number' at the end (e.g., 'chair_1', 'couch 2')
    # Captures: (_|\s)\d+$ 
    #   (_|\s): a preceding underscore or space
    #   \d+    : one or more digits
    #   $      : end of string
    label = re.sub(r'(_|\s)\d+$', '', label)
    
    # Regex 2: Remove '_char' or ' char' (single uppercase or lowercase letter) at the end 
    # Captures: (_|\s)[a-zA-Z]$
    # This is useful for formats like 'table_A'
    label = re.sub(r'(_|\s)[a-zA-Z]$', '', label)
    
    return label.strip()

def are_labels_similar_clip(label1: str, label2: str, similarity_threshold: float) -> bool:
    """
    Determines if two semantic labels are similar using CLIP cosine similarity.
    """
    if not CLIP_MODEL:
        # Fallback if initialization failed
        print("CLIP model not available, falling back to exact string match.")
        return label1.lower() == label2.lower()

    # 1. Get embeddings
    embedding1 = get_text_embedding(label1)
    embedding2 = get_text_embedding(label2)

    # 2. Calculate cosine similarity
    # Cosine similarity is a dot product of two normalized vectors
    similarity = torch.matmul(embedding1, embedding2.T).item()
    
    # 3. Compare with threshold
    return similarity >= similarity_threshold

# --- Helper Functions (From Previous Script) ---

def calculate_3d_iou(bbox1_bounds, bbox2_bounds):
    """Calculates the 3D Intersection over Union (IoU) of two AABBs."""
    min_A = np.array(bbox1_bounds[0])
    max_A = np.array(bbox1_bounds[1])
    min_B = np.array(bbox2_bounds[0])
    max_B = np.array(bbox2_bounds[1])

    intersect_min = np.maximum(min_A, min_B)
    intersect_max = np.minimum(max_A, max_B)

    intersect_dims = np.maximum(0, intersect_max - intersect_min)
    intersect_volume = np.prod(intersect_dims)

    volume_A = np.prod(max_A - min_A)
    volume_B = np.prod(max_B - min_B)

    if volume_A <= 1e-6 and volume_B <= 1e-6:
        return 1.0
    if volume_A <= 1e-6 or volume_B <= 1e-6:
        return 0.0

    union_volume = volume_A + volume_B - intersect_volume
    
    if union_volume == 0:
        return 0.0

    iou = intersect_volume / union_volume
    return iou

def check_overlaps(scene_graph: dict, iou_threshold: float) -> list:
    """Checks for overlaps between unique objects in the scene graph."""
    objects = list(scene_graph.items())
    overlaps = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            id1, data1 = objects[i]
            id2, data2 = objects[j]
            bounds1 = data1["obj_in_world_info"]["aabb_bounds"]
            bounds2 = data2["obj_in_world_info"]["aabb_bounds"]
            iou = calculate_3d_iou(bounds1, bounds2)
            if iou > iou_threshold:
                overlaps.append({
                    "obj1": id1,
                    "obj2": id2,
                    "iou": iou
                })
    return overlaps

# --- Main Deduplication Function (Updated to use CLIP) ---

def deduplicate_scene_graph(input_data: dict, iou_threshold: float, clip_threshold: float) -> dict:
    """
    Flattens the step-wise scene graph and deduplicates objects based on
    CLIP label similarity and 3D IoU overlap.
    """
    
    # 1. Flatten the scene graph into a list of all detected objects
    all_objects = []
    for step_id, objects_in_step in input_data.items():
        for label, data in objects_in_step.items():
            obj_data = {
                "step_id": step_id,
                "original_label": label,
                "confidence": data.get("confidence", 0.0),
                "aabb_bounds": data["obj_in_world_info"]["aabb_bounds"],
                "data": data 
            }
            all_objects.append(obj_data)

    # 2. Group objects into unique instances
    unique_groups = []
    
    for current_obj in all_objects:
        is_matched = False
        
        for group in unique_groups:
            # Compare the current object against the best representative (highest confidence)
            representative = group[0] 
            
            # Check for label similarity using CLIP
            label_match = are_labels_similar_clip(
                current_obj["original_label"], 
                representative["original_label"], 
                clip_threshold
            )

            if label_match:
                # Check for geometric overlap (IoU)
                iou = calculate_3d_iou(current_obj["aabb_bounds"], representative["aabb_bounds"])

                if iou >= iou_threshold:
                    # Match found! Add the current object to this group.
                    group.append(current_obj)
                    
                    # Update the representative if the current object has higher confidence
                    if current_obj["confidence"] > representative["confidence"]:
                        # Swap the representative to be the object with the highest confidence
                        group.insert(0, group.pop(-1)) 
                        
                    is_matched = True
                    break
        
        if not is_matched:
            # No match found, start a new unique group
            unique_groups.append([current_obj])

    # 3. Create the consolidated scene graph
    consolidated_scene_graph = {}
    for idx, group in enumerate(unique_groups):
        best_obj = group[0]
        final_label = best_obj["original_label"].replace(" ", "_").lower()
        unique_id = f"{final_label}_{uuid.uuid4().hex[:4]}"
        
        consolidated_scene_graph[unique_id] = best_obj["data"]
        consolidated_scene_graph[unique_id]["label"] = final_label

    return consolidated_scene_graph

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to dataset")
    parser.add_argument("lidar_dir", type=str, help="Input scene graph JSON file name")
    parser.add_argument("--iou_threshold", type=float, default=0.2, help="IoU threshold for deduplication")
    parser.add_argument("--clip_threshold", type=float, default=0.8, help="CLIP similarity threshold for deduplication")
    args = parser.parse_args()

    initialize_clip_model()

    input_file = os.path.join(args.dataset_dir, "scene_graph.json")
    output_file = os.path.join(args.dataset_dir, "scene_graph_reduced.json")
    with open(input_file, 'r') as f:
        scene_data = json.load(f)

    print(f"Deduplicating scene graph with IoU threshold: {args.iou_threshold}...")
    deduplicated_data = deduplicate_scene_graph(scene_data, args.iou_threshold, args.clip_threshold)

    with open(output_file, 'w') as f:
        json.dump(deduplicated_data, f, indent=2)

    print("Deduplication complete.")
            
    # Check for overlaps in deduplicated scene graph
    overlaps = check_overlaps(deduplicated_data, args.iou_threshold)
    if overlaps:
        print(f"Found {len(overlaps)} overlaps greater than IoU threshold {args.iou_threshold}:")
        for o in overlaps:
            print(f"  {o['obj1']} and {o['obj2']}: IoU {o['iou']:.3f}")
    else:
        print("No overlaps found in deduplicated scene graph.")
    # Plot CLIP embeddings
    if CLIP_MODEL:
        try:
            labels = [v.get("label", k.split('_')[0]) for k, v in deduplicated_data.items()]
            embeddings = [get_text_embedding(label) for label in labels]
            emb_tensor = torch.stack(embeddings).squeeze(1).cpu().numpy()
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(emb_tensor)
            plt.figure(figsize=(10, 8))
            for i, label in enumerate(labels):
                plt.scatter(emb_2d[i, 0], emb_2d[i, 1])
                plt.annotate(label, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8)
            plt.title(f"CLIP Embeddings PCA (Threshold: {args.clip_threshold})")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plot_path = os.path.join(args.dataset_dir, "clip_embeddings_plot.png")
            plt.savefig(plot_path)
            print(f"Saved CLIP embeddings plot to {plot_path}")
        except Exception as e:
            print(f"Failed to plot embeddings: {e}")
    else:
        print("CLIP model not available, skipping plot.")

    from demo_pc_rs import accum_pc
    accum_pc(args.dataset_dir, "scene_graph_reduced.json", args.lidar_dir)