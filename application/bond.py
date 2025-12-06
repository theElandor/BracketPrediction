"""
Modified testing script with postprocessing and visualization.
Command to run:
python application/bond.py \
    --config-file /homes/mlugli/BracketPrediction/application/configs/Pt_regressor_app.py \
    --options data_folder=/homes/mlugli/BracketPrediction/application/data/2/ \
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
from matplotlib.lines import Line2D

from visualizers import plot_teeth
from visualizers import plot_jaw

def write_points_to_ply_with_colors(filename, incisal, outer, origin, xaxis, yaxis, zaxis):
    """Write points to PLY file with colors to distinguish groups."""
    
    # Define colors for each group (RGB, 0-255)
    colors = {
        'incisal': [255, 0, 0],      # Red
        'outer': [0, 255, 0],         # Green
        'origin': [0, 0, 255],        # Blue
        'xaxis': [255, 255, 0],       # Yellow
        'yaxis': [255, 0, 255],       # Magenta
        'zaxis': [0, 255, 255]        # Cyan
    }
    
    all_points = []
    all_colors = []
    
    for name, point_array in [('incisal', incisal), ('outer', outer), 
                               ('origin', origin), ('xaxis', xaxis), 
                               ('yaxis', yaxis), ('zaxis', zaxis)]:
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, -1)
        
        all_points.append(point_array)
        # Repeat color for each point in the group
        all_colors.extend([colors[name]] * len(point_array))
    
    points = np.vstack(all_points)
    colors_array = np.array(all_colors)
    num_points = points.shape[0]
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for point, color in zip(points, colors_array):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

def process_tooth_predictions(mesh, bracket_pred, incisal_pred, outer_pred, 
                                patient_id, fdi, output_dir, tooth_key, teeth_path, visualize: bool = True):
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
        teeth_path: path to the teeth directory containing transformation JSON files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vertices = mesh.vertices
    
    # Load transformation parameters from tooth's JSON file
    transform_file = teeth_path / f"{tooth_key}.json"
    translation = np.array([0.0, 0.0, 0.0])
    scaling = 1.0

    with open(transform_file, 'r') as f:
        transform_data = json.load(f)
        translation = np.array(transform_data.get('translation', [0.0, 0.0, 0.0]))
        scaling = float(transform_data.get('scaling', 1.0))
        print(f"  Loaded transformation for {tooth_key}: translation={translation}, scaling={scaling}")

    # ========= Project point on surface =============
    predictions = [bracket_pred, incisal_pred, outer_pred]
    face_ids = []
    projected_points = []
    for pred in predictions:
        closest_points, distance, faces = mesh.nearest.on_surface([np.array(pred)])
        face_ids.append(faces)
        projected_points.append(closest_points[0])
    bracket, incisal, outer = projected_points
    bracket_face_id, _, _ = face_ids

    json_data = None
    v_normal = mesh.face_normals[bracket_face_id[0]]
    v_normal = v_normal / np.linalg.norm(v_normal)

    # Get axis between incisal and outer points
    v_io = outer - incisal
    v_io = v_io / np.linalg.norm(v_io)

    # Get perpendicular axis to define plane
    v_perp = np.cross(v_normal, v_io)
    v_perp = v_perp / np.linalg.norm(v_perp)
    
    # Apply inverse transformation: denormalize the points
    bracket_denorm = bracket / scaling + translation
    incisal_denorm = incisal / scaling + translation
    outer_denorm = outer / scaling + translation
    v_perp_denorm = v_perp / scaling
    v_normal_denorm = v_normal / scaling
    
    json_data = {
        "incisal": incisal_denorm.tolist(),
        "outer": outer_denorm.tolist(),
        "basePlane": {
            "origin": bracket_denorm.tolist(),
            "xAxis": incisal_denorm.tolist(),
            "yAxis": (bracket_denorm + v_perp_denorm).tolist(),
            "zAxis": (bracket_denorm + v_normal_denorm).tolist(),
        },
    }
    if visualize:
        plot_teeth(bracket, incisal, outer, v_io, v_perp,
                   vertices, patient_id, fdi, output_dir)
    return json_data

def postprocess_predictions(data_folder, visualize: bool = True):
    """
    Post-processes predictions and creates visualizations.
    
    Args:
        data_folder: Path to the data folder containing predictions
        visualize: toggles visualization
    """
    data_folder = Path(data_folder)
    output_reg_path = data_folder / "output_reg" / "results"
    teeth_path = data_folder / "output_seg" / "teeth"
    viz_dir = data_folder /  "output_reg" / "plots"
    print("\n=== Starting Post-Processing and Visualization ===")
    pred_files = list(output_reg_path.glob("*.json"))
    if not pred_files:
        print(f"No prediction files found in {output_reg_path}")
        return    
    pred_file = pred_files[0]
    print(f"Loading predictions from: {pred_file.name}")
    with open(pred_file, 'r') as f: all_predictions = json.load(f)
    print(f"Found predictions for {len(all_predictions)} teeth")    
    all_points_data = {}
    for tooth_key, predictions in all_predictions.items():
        # Parse tooth_key: expected format "STEM_lower_0002_FDI_47"
        parts = tooth_key.split('_')
        patient_idx = parts.index('lower') if 'lower' in parts else parts.index('upper')
        patient_id = parts[patient_idx + 1]
        fdi_idx = parts.index('FDI')
        fdi = int(parts[fdi_idx + 1])
        stl_file = teeth_path / f"{tooth_key}.stl"
        mesh = trimesh.load_mesh(stl_file)
 
        # Get predictions
        bracket_pred = predictions.get('bracket')
        incisal_pred = predictions.get('incisal')
        outer_pred = predictions.get('outer')
 
        # Process single tooth predictions to
        # bring them back to normalized coordinates
        points_data = process_tooth_predictions(
            mesh=mesh,
            bracket_pred=bracket_pred,
            incisal_pred=incisal_pred,
            outer_pred=outer_pred,
            patient_id=patient_id,
            fdi=fdi,
            output_dir=viz_dir,
            tooth_key=tooth_key,
            teeth_path=teeth_path,
            visualize=visualize
        )
        all_points_data[tooth_key] = points_data

    # Save all points to a single JSON file
    output_json_path = output_reg_path / "projected_points.json"
    with open(output_json_path, "w") as f: json.dump(all_points_data, f, indent=4)
    print(f"\n💾 Saved all projected points to: {output_json_path}")

    # Rotated version of points file.
    rotated_points = {}
    all_incisal, all_outer, all_origin, all_xaxis, all_yaxis, all_zaxis = [], [], [], [], [], []
    all_incisal_pre, all_outer_pre, all_origin_pre, all_xaxis_pre, all_yaxis_pre, all_zaxis_pre = [], [], [], [], [], []
    for tooth_key, pdata in all_points_data.items():
        # Load shift file before rotation
        shift = np.array([0.0, 0.0, 0.0])
        jaw_type = 'lower' if 'lower' in tooth_key else 'upper'
        shift_file_name = f"STEM_{jaw_type}_{patient_id}_shift.json"
        shift_file = data_folder.parent / f"{patient_id}" / shift_file_name

        if shift_file.exists():
            try:
                with open(shift_file, 'r') as f:
                    shift_data = json.load(f)
                    shift = np.array(shift_data.get('shift', [0.0, 0.0, 0.0]))
            except Exception as e:
                print(f"  ⚠️ Could not load or parse shift file {shift_file}: {e}")

        # Choose rotation sequence based on jaw
        seq = [('x', -90), ('y', 180)]
        try:
            # Apply shift to points, then apply rotations using trimesh
            incisal = np.array(pdata['incisal']) + shift
            outer = np.array(pdata['outer']) + shift
            origin = np.array(pdata['basePlane']['origin']) + shift
            xaxis = np.array(pdata['basePlane']['xAxis']) + shift
            yaxis = np.array(pdata['basePlane']['yAxis']) + shift
            zaxis = np.array(pdata['basePlane']['zAxis']) + shift

            all_incisal_pre.append(incisal)
            all_outer_pre.append(outer)
            all_origin_pre.append(origin)
            all_xaxis_pre.append(xaxis)
            all_yaxis_pre.append(yaxis)
            all_zaxis_pre.append(zaxis)
            
            # Apply rotation sequence using trimesh transformations
            for axis, degrees in seq:
                radians = np.deg2rad(degrees)
                if axis == 'x':
                    rotation = trimesh.transformations.rotation_matrix(radians, [1, 0, 0])
                elif axis == 'y':
                    rotation = trimesh.transformations.rotation_matrix(radians, [0, 1, 0])
                else:
                    rotation = trimesh.transformations.rotation_matrix(radians, [0, 0, 1])
                
                # Apply transformation to points
                incisal = trimesh.transformations.transform_points([incisal], rotation)[0]
                outer = trimesh.transformations.transform_points([outer], rotation)[0]
                origin = trimesh.transformations.transform_points([origin], rotation)[0]
                xaxis = trimesh.transformations.transform_points([xaxis], rotation)[0]
                yaxis = trimesh.transformations.transform_points([yaxis], rotation)[0]
                zaxis = trimesh.transformations.transform_points([zaxis], rotation)[0]

            rotated_points[tooth_key] = {
                'incisal': incisal.tolist(),
                'outer': outer.tolist(),
                'basePlane': {
                    'origin': origin.tolist(),
                    'xAxis': xaxis.tolist(),
                    'yAxis': yaxis.tolist(),
                    'zAxis': zaxis.tolist(),
                }
            }
            all_incisal.append(incisal)
            all_outer.append(outer)
            all_origin.append(origin)
            all_xaxis.append(xaxis)
            all_yaxis.append(yaxis)
            all_zaxis.append(zaxis)
        except Exception as e:
            print(f"⚠️ Error rotating points for {tooth_key}: {e}")

    # Save all points to a single PLY file (pre-rotation)
    output_ply_path_pre = output_reg_path / "all_keypoints_pre_rotation.ply"
    write_points_to_ply_with_colors(
        str(output_ply_path_pre),
        np.array(all_incisal_pre),
        np.array(all_outer_pre),
        np.array(all_origin_pre),
        np.array(all_xaxis_pre),
        np.array(all_yaxis_pre),
        np.array(all_zaxis_pre),
    )
    print(f"\n💾 Saved all pre-rotation keypoints to: {output_ply_path_pre}")
            
    # Save all points to a single PLY file
    output_ply_path = output_reg_path / "all_keypoints.ply"
    write_points_to_ply_with_colors(
        str(output_ply_path),
        np.array(all_incisal),
        np.array(all_outer),
        np.array(all_origin),
        np.array(all_xaxis),
        np.array(all_yaxis),
        np.array(all_zaxis),
    )
    print(f"\n💾 Saved all keypoints to: {output_ply_path}")

    rotated_output_path = output_reg_path / "projected_points_rotated.json"
    with open(rotated_output_path, 'w') as f: json.dump(rotated_points, f, indent=4)
    print(f"\n💾 Saved rotated projected points to: {rotated_output_path}")
    print(f"\n✅ Post-processing complete. Visualizations saved to: {viz_dir}")
    
    # ============= Debug visualizations ==================
    if visualize:
        plot_jaw(data_folder, raw_scan=False)
        plot_jaw(data_folder, raw_scan=True)

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
    postprocess_predictions(data_folder, visualize=not cfg.no_visuals)


def main():
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
    cfg._cfg_dict["data_root"] = str(Path(args.options["data_folder"]) /  "output_seg" / "teeth")
    cfg._cfg_dict["save_path"] = str(Path(args.options["data_folder"]) / "output_reg") 
    cfg._cfg_dict["data"]["test"]["data_root"] = str(Path(args.options["data_folder"]) /  "output_seg" / "teeth")
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
    main()