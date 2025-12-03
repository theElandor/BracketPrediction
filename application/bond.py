"""
Modified testing script with postprocessing and visualization.
Command to run:
python application/bond.py \
    --config-file /homes/mlugli/BracketPrediction/application/configs/Pt_regressor_app.py \
    --options data_folder=/homes/mlugli/BracketPrediction/application/data/0002/ \
              weight=/homes/mlugli/BracketPrediction/application/weights/regressor_best.pth
"""

import debugpy
import os
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def visualize_tooth_predictions(mesh, bracket_pred, incisal_pred, outer_pred, 
                                patient_id, fdi, output_dir, tooth_key):
    """
    Creates three 2D views (XY, XZ, YZ) of the mesh with predicted points.
    Projects predictions onto mesh surface using nearest point method. 
    Args:
        mesh: trimesh object
        bracket_pred: bracket point coordinates [x, y, z]
        incisal_pred: incisal point coordinates [x, y, z]
        outer_pred: outer point coordinates [x, y, z]
        patient_id: patient identifier
        fdi: FDI tooth index
        output_dir: directory to save the PNG file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vertices = mesh.vertices
    
    # Create figure with 3 subplots for different views
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert predictions to numpy arrays and project onto mesh
    bracket_original = np.array(bracket_pred)
    closest_point_array, distance, bracket_face_id = mesh.nearest.on_surface([bracket_original])
    bracket = closest_point_array[0]
    
    incisal = None
    if incisal_pred is not None:
        incisal_original = np.array(incisal_pred)
        closest_point_array, _, _ = mesh.nearest.on_surface([incisal_original])
        incisal = closest_point_array[0]
    
    outer = None
    if outer_pred is not None:
        outer_original = np.array(outer_pred)
        closest_point_array, _, _ = mesh.nearest.on_surface([outer_original])
        outer = closest_point_array[0]

    json_data = None
    # Plot plane if incisal and outer points are available
    if incisal is not None and outer is not None:
        # Get mesh normal at bracket point
        v_normal = mesh.face_normals[bracket_face_id[0]]
        if np.linalg.norm(v_normal) > 1e-6:
            v_normal = v_normal / np.linalg.norm(v_normal)

            # Get axis between incisal and outer points
            v_io = outer - incisal
            if np.linalg.norm(v_io) > 1e-6:
                v_io = v_io / np.linalg.norm(v_io)

                # Get perpendicular axis to define plane
                v_perp = np.cross(v_normal, v_io)
                if np.linalg.norm(v_perp) > 1e-6:
                    v_perp = v_perp / np.linalg.norm(v_perp)
                    json_data = {
                        "incisal": incisal.tolist(),
                        "outer": outer.tolist(),
                        "basePlane": {
                            "origin": bracket.tolist(),
                            "xAxis": incisal.tolist(),
                            "yAxis": (bracket + v_perp).tolist(),
                            "zAxis": (bracket + v_normal).tolist(),
                        },
                    }
                    
                    # Create plane points
                    plane_size = 0.5  # smaller
                    plane_res = 30  # denser
                    s = np.linspace(-plane_size, plane_size, plane_res)
                    t = np.linspace(-plane_size, plane_size, plane_res)
                    ss, tt = np.meshgrid(s, t)
                    
                    plane_points = bracket + ss[..., None] * v_io + tt[..., None] * v_perp
                    plane_points = plane_points.reshape(-1, 3)

                    # Plot the plane on all three views
                    axes[0].scatter(plane_points[:, 0], plane_points[:, 1], c='gray', s=5, alpha=0.5, label='Plane')
                    axes[1].scatter(plane_points[:, 0], plane_points[:, 2], c='gray', s=5, alpha=0.5)
                    axes[2].scatter(plane_points[:, 1], plane_points[:, 2], c='gray', s=5, alpha=0.5)
    
    # View 1: XY plane (looking down Z-axis)
    ax = axes[0]
    ax.scatter(vertices[:, 0], vertices[:, 1], c='lightblue', s=3, alpha=0.3, label='Mesh')
    ax.scatter(bracket[0], bracket[1], c='orange', s=100, marker='o', 
              edgecolors='black', linewidths=2, label='Bracket', zorder=5)
    if incisal is not None:
        ax.scatter(incisal[0], incisal[1], c='cyan', s=100, marker='s', 
                  edgecolors='black', linewidths=2, label='Incisal', zorder=5)
    if outer is not None:
        ax.scatter(outer[0], outer[1], c='salmon', s=100, marker='^', 
                  edgecolors='black', linewidths=2, label='Outer', zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('XY View (Top)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # View 2: XZ plane (looking from Y-axis)
    ax = axes[1]
    ax.scatter(vertices[:, 0], vertices[:, 2], c='lightblue', s=3, alpha=0.3, label='Mesh')
    ax.scatter(bracket[0], bracket[2], c='orange', s=100, marker='o', 
              edgecolors='black', linewidths=2, label='Bracket', zorder=5)
    if incisal is not None:
        ax.scatter(incisal[0], incisal[2], c='cyan', s=100, marker='s', 
                  edgecolors='black', linewidths=2, label='Incisal', zorder=5)
    if outer is not None:
        ax.scatter(outer[0], outer[2], c='salmon', s=100, marker='^', 
                  edgecolors='black', linewidths=2, label='Outer', zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('XZ View (Front)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # View 3: YZ plane (looking from X-axis)
    ax = axes[2]
    ax.scatter(vertices[:, 1], vertices[:, 2], c='lightblue', s=3, alpha=0.3, label='Mesh')
    ax.scatter(bracket[1], bracket[2], c='orange', s=100, marker='o', 
              edgecolors='black', linewidths=2, label='Bracket', zorder=5)
    if incisal is not None:
        ax.scatter(incisal[1], incisal[2], c='cyan', s=100, marker='s', 
                  edgecolors='black', linewidths=2, label='Incisal', zorder=5)
    if outer is not None:
        ax.scatter(outer[1], outer[2], c='salmon', s=100, marker='^', 
                  edgecolors='black', linewidths=2, label='Outer', zorder=5)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('YZ View (Side)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Patient {patient_id} - Tooth FDI {fdi}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"patient_{patient_id}_FDI_{fdi}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved visualization: {output_file}")
    return json_data


def postprocess_predictions(data_folder):
    """
    Post-processes predictions and creates visualizations.
    
    Args:
        data_folder: Path to the data folder containing predictions
    """
    data_folder = Path(data_folder)
    output_reg_path = data_folder / "output_reg" / "results"
    teeth_path = data_folder / "output_seg" / "teeth"
    viz_dir = data_folder /  "output_reg" / "plots"
    
    print("\n=== Starting Post-Processing and Visualization ===")
    
    # Find the predictions file (should be a single JSON with all predictions)
    pred_files = list(output_reg_path.glob("*.json"))
    
    if not pred_files:
        print(f"No prediction files found in {output_reg_path}")
        return
    
    if len(pred_files) > 1:
        print(f"⚠️ Found multiple prediction files, using the first one: {pred_files[0].name}")
    
    pred_file = pred_files[0]
    print(f"Loading predictions from: {pred_file.name}")
    
    # Load all predictions
    try:
        with open(pred_file, 'r') as f:
            all_predictions = json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading predictions file: {e}")
        return
    
    print(f"Found predictions for {len(all_predictions)} teeth")
    
    all_points_data = {}
    # Iterate through each tooth in the predictions
    for tooth_key, predictions in all_predictions.items():
        # Parse tooth_key: expected format "STEM_lower_0002_FDI_47"
        parts = tooth_key.split('_')
        
        try:
            # Find patient_id and FDI from key
            patient_idx = parts.index('lower') if 'lower' in parts else parts.index('upper')
            patient_id = parts[patient_idx + 1]
            fdi_idx = parts.index('FDI')
            fdi = int(parts[fdi_idx + 1])
        except (ValueError, IndexError):
            print(f"⚠️ Could not parse tooth key: {tooth_key}, skipping")
            continue
        
        # Find corresponding STL file
        stl_file = teeth_path / f"{tooth_key}.stl"
        if not stl_file.exists():
            print(f"⚠️ Missing STL file for {tooth_key}, skipping visualization")
            continue
        
        # Load mesh
        try:
            mesh = trimesh.load_mesh(stl_file)
        except Exception as e:
            print(f"⚠️ Error loading mesh {tooth_key}.stl: {e}")
            continue
        
        # Extract predictions for this tooth
        bracket_pred = predictions.get('bracket')
        incisal_pred = predictions.get('incisal')
        outer_pred = predictions.get('outer')
        
        if bracket_pred is None:
            print(f"⚠️ No bracket prediction for {tooth_key}, skipping")
            continue
        
        # Create visualization
        try:
            points_data = visualize_tooth_predictions(
                mesh=mesh,
                bracket_pred=bracket_pred,
                incisal_pred=incisal_pred,
                outer_pred=outer_pred,
                patient_id=patient_id,
                fdi=fdi,
                output_dir=viz_dir,
                tooth_key=tooth_key
            )
            if points_data:
                all_points_data[tooth_key] = points_data
        except Exception as e:
            print(f"⚠️ Error creating visualization for {tooth_key}: {e}")
            continue
    
    # Save all points to a single JSON file
    if all_points_data:
        output_json_path = output_reg_path / "projected_points.json"
        try:
            with open(output_json_path, "w") as f:
                json.dump(all_points_data, f, indent=4)
            print(f"\n💾 Saved all projected points to: {output_json_path}")
        except Exception as e:
            print(f"⚠️ Error saving aggregated JSON file: {e}")
    
    print(f"\n✅ Post-processing complete. Visualizations saved to: {viz_dir}")



def main_worker(cfg):
    os.makedirs(cfg.save_path, exist_ok=True)
    cfg = default_setup(cfg)
    test_cfg = dict(cfg=cfg, **cfg.test)
    tester = TESTERS.build(test_cfg)
    tester.test()
    
    # Add post-processing and visualization after testing
    print("\n" + "="*60)
    print("Testing complete. Starting post-processing...")
    print("="*60)
    
    # Extract data_folder from save_path
    data_folder = Path(cfg.save_path).parent
    postprocess_predictions(data_folder)


def main():
    args = default_argument_parser().parse_args()
    if args.debug == True:
        print("Hello, happy debugging.")
        debugpy.listen(("0.0.0.0", 5681))
        print(">>> Debugger is listening on port 5681. Waiting for client to attach...")
        debugpy.wait_for_client()
        print(">>> Debugger attached. Resuming execution.")
    cfg = default_config_parser(args.config_file, args.options)
    cfg._cfg_dict["data_root"] = str(Path(args.options["data_folder"]) /  "output_seg" / "teeth")
    cfg._cfg_dict["save_path"] = str(Path(args.options["data_folder"]) / "output_reg") 
    cfg._cfg_dict["data"]["test"]["data_root"] = str(Path(args.options["data_folder"]) /  "output_seg" / "teeth")
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()