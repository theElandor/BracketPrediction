"""
Modified testing script with postprocessing.
Command to run:

python application/segment_scan.py \
    --config-file /homes/mlugli/BracketPrediction/application/configs/Pt_semseg_app.py \
    --options data_folder=/homes/mlugli/BracketPrediction/application/data/0002/ \
              weight=/homes/mlugli/BracketPrediction/application/weights/segmentator_best.pth
"""
MAPPING = {
    1: 48, 2: 47, 3: 46,
    4: 45, 5: 44, 6: 43,
    7: 42, 8: 41, 9: 31,
    10: 32, 11: 33, 12: 34,
    13: 35, 14: 36, 15: 37,
    16: 38
}
import debugpy
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pathlib import Path
from pointcept.engines.launch import launch
import numpy as np
import pyvista as pv
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import os


def normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize points to unit sphere centered at origin.
    Returns: (normalized_points, translation, scale)
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    distances = np.linalg.norm(centered_points, axis=1)
    max_distance = np.max(distances)
    scale = 1.0 / max_distance if max_distance > 0 else 1.0
    normalized_points = centered_points * scale
    return normalized_points, centroid, scale


def create_segmentation_visualization(mesh, mask, name, output_dir: Path):
    """
    Create a visualization of the segmented dental arch from 3 viewpoints.
    
    Args:
        stl_file: Path to original STL file
        mask_file: Path to predicted segmentation mask (.npy)
        output_dir: Output directory for visualization
    """
    # Load mesh and mask    
    if len(mask) != len(mesh.points):
        print(f"Warning: Cannot visualize - mask length mismatch")
        return
    
    # Assign colors based on FDI index
    unique_fdi = np.unique(mask)
    cmap = cm.get_cmap('tab20')
    
    # Create color array for vertices
    colors = np.zeros((len(mask), 3))
    for i, fdi_val in enumerate(unique_fdi):
        if fdi_val == 0:  # Gum - use gray
            colors[mask == fdi_val] = [0.7, 0.7, 0.7]
        else:
            rgb = cmap((i % 20) / 20.0)[:3]
            colors[mask == fdi_val] = rgb
    
    mesh['colors'] = colors
    
    # Define three viewpoints
    viewpoints = [
        {'azimuth': 0, 'elevation': 0, 'title': 'Front'},      # Front view
        {'azimuth': 90, 'elevation': 0, 'title': 'Side'},      # Side view
        {'azimuth': 0, 'elevation': 90, 'title': 'Top'}        # Top view
    ]
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    
    for idx, vp in enumerate(viewpoints):
        plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
        plotter.add_mesh(mesh, scalars='colors', rgb=True, lighting=False)
        plotter.camera.azimuth = vp['azimuth']
        plotter.camera.elevation = vp['elevation']
        plotter.camera.zoom(1.3)
        img = plotter.screenshot(return_img=True)
        plotter.close()
        ax = fig.add_subplot(1, 3, idx + 1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(vp['title'], fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Segmentation: {name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    vis_output_path = output_dir / f"{name}_segmentation_views.png"
    plt.savefig(vis_output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {vis_output_path}")


def postprocess_segmentation(stl_file: Path, mask_file: Path, output_dir: Path, visualize: bool = True):
    """
    Postprocess segmentation results: split by tooth, normalize, and save.
    
    Args:
        stl_file: Path to original STL file
        mask_file: Path to predicted segmentation mask (.npy)
        output_dir: Output directory for processed teeth
    """
    print(f"Postprocessing {stl_file.name}...")
    
    # Load mesh and mask
    mesh = pv.read(stl_file)
    mask = np.load(mask_file)
    if visualize:
        create_segmentation_visualization(mesh, mask, stl_file.stem, output_dir) 
    if len(mask) != len(mesh.points):
        print(f"Warning: Mask length ({len(mask)}) doesn't match points ({len(mesh.points)})")
        return
    
    base_name = stl_file.stem
    stl_output_dir = output_dir / "teeth"
    json_output_dir = output_dir / "teeth"
    stl_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique FDI indices (excluding 0 which is gum)
    unique_fdi_indices = np.unique(mask)
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Remove the '3' prefix from each face
    print(f"Found {len(unique_fdi_indices)} unique classes: {unique_fdi_indices}")

    # Process teeth 
    for fdi_index in unique_fdi_indices:
        if fdi_index == 0:
            continue  # Skip gum
        
        # Create folder for this FDI index
        # Get points belonging to this tooth
        class_mask = mask == fdi_index
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) == 0:
            continue
        
        # Extract points for this class
        class_points = points[class_mask]
        
        # Normalize points
        normalized_class_points, translation, scale = normalize(class_points)
        
        # Find faces that have all vertices in this class
        face_mask = np.all(np.isin(faces, class_indices), axis=1)
        class_faces_old_idx = faces[face_mask]
        
        # Remap face indices to new point array
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
        class_faces = np.array([[old_to_new[idx] for idx in face] for face in class_faces_old_idx])
        
        # Create PyVista mesh for this tooth
        # Faces need to be in format: [3, v0, v1, v2, 3, v3, v4, v5, ...]
        faces_pv = np.hstack([[3] + list(face) for face in class_faces])
        tooth_mesh = pv.PolyData(normalized_class_points, faces_pv)
 
        # Save STL file
        if "lower" in base_name: fdi_index = MAPPING[fdi_index]
        if "upper" in base_name: fdi_index = MAPPING[fdi_index]-20
        stl_output_path = stl_output_dir / f"{base_name}_FDI_{fdi_index}.stl"
        tooth_mesh.save(stl_output_path)
        
        # Save normalization parameters to JSON
        json_output_path = json_output_dir / f"{base_name}_FDI_{fdi_index}.json"
        json_data = {
            "translation": translation.tolist(),
            "scaling": float(scale)
        }
        
        with open(json_output_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"  Saved FDI {fdi_index}: {len(class_points)} points, {len(class_faces)} faces")
        print(f"    STL: {stl_output_path}")
        print(f"    JSON: {json_output_path}")


def main_worker(cfg):
    os.makedirs(cfg.save_path, exist_ok=True)
    cfg = default_setup(cfg)
    test_cfg = dict(cfg=cfg, **cfg.test)
    tester = TESTERS.build(test_cfg)
    tester.test()
    
    # Postprocessing: split and normalize teeth
    print("\n" + "="*80)
    print("Starting postprocessing...")
    print("="*80 + "\n")
    
    data_folder = Path(cfg.data_root)
    output_folder = Path(cfg.save_path)
    visualize = not cfg.no_visuals
    
    # Find STL files in data folder
    stl_files = list(data_folder.glob("*.stl"))
    
    for stl_file in stl_files:
        # Find corresponding prediction mask
        mask_file = output_folder / "result" / f"{stl_file.stem}_pred.npy"
        
        if not mask_file.exists():
            print(f"Warning: No prediction found for {stl_file.name}, skipping...")
            continue
        
        postprocess_segmentation(stl_file, mask_file, output_folder, visualize=visualize)
    
    print("\n" + "="*80)
    print("Postprocessing complete!")
    print("="*80)


def segment_scan():
    parser = default_argument_parser()
    parser.add_argument("--no-visuals", action="store_true", help="Do not generate visualizations")
    args = parser.parse_args()

    if args.debug:
        print("Hello, happy debugging.")
        debugpy.listen(("0.0.0.0", 5681))
        print(">>> Debugger is listening on port 5681. Waiting for client to attach...")
        debugpy.wait_for_client()
        print(">>> Debugger attached. Resuming execution.")
    cfg = default_config_parser(args.config_file, args.options)
    cfg._cfg_dict["data_root"] = args.options["data_folder"]
    cfg._cfg_dict["save_path"] = str(Path(args.options["data_folder"]) / "output_seg") 
    cfg._cfg_dict["data"]["test"]["data_root"] = args.options["data_folder"]
    cfg.no_visuals = args.no_visuals
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    segment_scan()
    print("Finished.")