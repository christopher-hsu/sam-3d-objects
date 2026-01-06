#!/usr/bin/env python3
"""
Combine GLB meshes listed in per-step scene_graph files produced by the demo.

This script is combine-only: it does not perform inference. It expects a dataset
folder with `step_XXXX` subfolders that each contain `scene_graph_*.json` files
produced by the pipeline. It will:
  - create per-step combined GLBs at `glbs_world/scene_world_step_XXXX.glb`
    (prefers `glb_world_path` entries in the scene_graph, falls back to `glb_path`)
  - create a single combined GLB `glbs_world/scene_world_all.glb` that contains
    all objects from all steps.

The script prefers `trimesh` for combining/exporting meshes. If `trimesh` is
not available it falls back to Open3D. Install `trimesh` for best results.
"""

import os
import json
import glob
import argparse
from typing import List

import numpy as np

try:
    import trimesh
except Exception:
    trimesh = None

try:
    import open3d as o3d
except Exception:
    o3d = None


def _read_dataset_scene_graphs(dataset_dir: str) -> dict:
    """Read central `scene_graph.json` if present, else gather per-step files.

    Returns a dict mapping step_name -> scene_graph (dict).
    """
    entries = {}
    central = os.path.join(dataset_dir, "scene_graph.json")
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


def _collect_glb_paths(scene_graphs: dict) -> List[str]:
    """Collect GLB file paths from loaded scene_graphs.

    For each object entry prefer `glb_world_path` if present and exists,
    otherwise fall back to `glb_path`.
    """
    paths = []
    for step, sg in scene_graphs.items():
        if not isinstance(sg, dict):
            continue
        for obj_name, obj in sg.items():
            gw = obj.get("glb_world_path")
            g = obj.get("glb_path")
            if gw and os.path.exists(gw):
                paths.append(gw)
            elif g and os.path.exists(g):
                paths.append(g)

    # deduplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _combine_meshes(mesh_paths: List[str], out_path: str) -> bool:
    """Combine meshes into a single file at `out_path`.

    Returns True on success.
    """
    if len(mesh_paths) == 0:
        print("No meshes to combine")
        return False

    # Prefer trimesh
    if trimesh is not None:
        meshes = []
        for p in mesh_paths:
            try:
                m = trimesh.load(p, force="mesh")
                meshes.append(m)
            except Exception as e:
                print(f"trimesh failed to load {p}: {e}")
        if len(meshes) == 0:
            print("trimesh could not load any meshes")
        else:
            try:
                combined = trimesh.util.concatenate(meshes)
                combined.export(out_path)
                return True
            except Exception as e:
                print(f"trimesh combine/export failed: {e}")

    # Fallback: Open3D
    if o3d is not None:
        verts_all = []
        faces_all = []
        vert_offset = 0
        for p in mesh_paths:
            try:
                m = o3d.io.read_triangle_mesh(p)
                v = np.asarray(m.vertices)
                f = np.asarray(m.triangles)
                if v.size == 0:
                    continue
                verts_all.append(v)
                faces_all.append(f + vert_offset)
                vert_offset += v.shape[0]
            except Exception as e:
                print(f"Open3D failed to load {p}: {e}")

        if len(verts_all) == 0:
            print("Open3D could not load any meshes")
        else:
            verts_comb = np.vstack(verts_all)
            faces_comb = np.vstack(faces_all)
            mesh_comb = o3d.geometry.TriangleMesh()
            mesh_comb.vertices = o3d.utility.Vector3dVector(verts_comb)
            mesh_comb.triangles = o3d.utility.Vector3iVector(faces_comb)
            try:
                o3d.io.write_triangle_mesh(out_path, mesh_comb)
                return True
            except Exception as e:
                print(f"Open3D export failed: {e}")

    print("No available backend succeeded (install trimesh or open3d)")
    return False


def combine_per_step(dataset_dir: str) -> None:
    """For each step folder, read its scene_graph and write a per-step combined GLB.

    Also writes a final `scene_world_all.glb` that concatenates all per-step meshes.
    """
    scene_graphs = _read_dataset_scene_graphs(dataset_dir)
    if len(scene_graphs) == 0:
        print("No scene_graphs found in dataset")
        return

    glb_world_dir = os.path.join(dataset_dir, "glbs_world")
    os.makedirs(glb_world_dir, exist_ok=True)

    all_paths = []
    # scene_graphs is a dict step->scene_graph
    for step, sg in sorted(scene_graphs.items()):
        if not isinstance(sg, dict):
            continue
        # collect glb paths for this step
        paths = []
        for obj_name, obj in sg.items():
            gw = obj.get("glb_world_path")
            g = obj.get("glb_path")
            if gw and os.path.exists(gw):
                paths.append(gw)
            elif g and os.path.exists(g):
                paths.append(g)

        if len(paths) == 0:
            print(f"No GLBs for step {step}, skipping")
            continue

        # write per-step combined GLB
        out_step = os.path.join(glb_world_dir, f"scene_world_{step}.glb")
        ok = _combine_meshes(paths, out_step)
        if ok:
            print(f"Wrote per-step combined GLB: {out_step}")
        else:
            print(f"Failed to write per-step combined GLB for {step}")

        all_paths.extend(paths)

    # combine everything into one GLB
    if len(all_paths) == 0:
        print("No GLB paths found across steps; nothing to combine")
        return

    out_all = os.path.join(glb_world_dir, "scene_world_all.glb")
    ok = _combine_meshes(all_paths, out_all)
    if ok:
        print(f"Wrote combined dataset GLB: {out_all}")
    else:
        print("Failed to write combined dataset GLB")


def main():
    parser = argparse.ArgumentParser(description="Combine GLBs from scene_graphs into per-step and dataset GLBs")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset root")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        print(f"Dataset dir not found: {dataset_dir}")
        return

    combine_per_step(dataset_dir)


if __name__ == "__main__":
    main()
