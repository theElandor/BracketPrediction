import json
import argparse
import sys
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from stl import mesh

def create_plots(json_file_path: str, stl_dir_paths: list):
    """
    Generates interactive 3D plots for STL files and corresponding points
    defined in a JSON file. Searches for STL files across multiple directories.
    """
    
    # --- 1. Validate paths and setup directories ---
    json_path = Path(json_file_path)
    stl_dirs = [Path(dir_path) for dir_path in stl_dir_paths]

    if not json_path.is_file():
        print(f"Error: JSON file not found at {json_path}")
        sys.exit(1)
    
    # Validate all STL directories
    for stl_dir in stl_dirs:
        if not stl_dir.is_dir():
            print(f"Error: STL directory not found at {stl_dir}")
            sys.exit(1)
    
    print(f"Searching for STL files in {len(stl_dirs)} directories:")
    for stl_dir in stl_dirs:
        print(f"  - {stl_dir.resolve()}")

    # Create the output directory "plots" in the same folder as the JSON file
    output_dir = json_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    print(f"Plots will be saved to: {output_dir.resolve()}")

    # --- 2. Load JSON data ---
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file {json_path}: {e}")
        sys.exit(1)
    
    print(f"Found {len(data)} entries in the JSON file.")

    # --- 3. Iterate, process, and plot ---
    processed_count = 0
    skipped_count = 0
    for filename_key, coordinates in data.items():
        
        # Construct the expected STL file name
        stl_file_name = f"{filename_key}.stl"
        
        # Search for the STL file in all provided directories
        stl_path = None
        for stl_dir in stl_dirs:
            candidate_path = stl_dir / stl_file_name
            if candidate_path.exists():
                stl_path = candidate_path
                break
        
        # Check if the STL file was found
        if stl_path is None:
            print(f"Warning: STL file '{stl_file_name}' not found in any directory. Skipping '{filename_key}'.")
            skipped_count += 1
            continue

        print(f"Processing: {filename_key} (found in {stl_path.parent.name}) ...")

        try:
            # --- 4. Load STL mesh ---
            your_mesh = mesh.Mesh.from_file(stl_path)

            # --- 5. Extract data for Plotly's Mesh3d trace ---
            # Plotly's Mesh3d trace requires vertices (x, y, z) and
            # face indices (i, j, k).
            # We can flatten the vectors from numpy-stl to get all vertices
            # for all triangles.
            x = your_mesh.x.flatten()
            y = your_mesh.y.flatten()
            z = your_mesh.z.flatten()

            # Create the indices for the triangles
            # For this flattened structure, the triangles are just
            # (0, 1, 2), (3, 4, 5), (6, 7, 8), and so on.
            total_points = len(x)
            i = np.arange(0, total_points, 3)
            j = np.arange(1, total_points, 3)
            k = np.arange(2, total_points, 3)

            # --- 6. Create Plotly traces ---
            
            # The 3D Mesh trace
            mesh_trace = go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.7,
                color='gray',
                name=filename_key,
                hoverinfo='name'
            )

            # The 3D Scatter plot trace for the single point
            point_trace = go.Scatter3d(
                x=[coordinates[0]],
                y=[coordinates[1]],
                z=[coordinates[2]],
                mode='markers',
                marker=dict(
                    size=10,         # Set a clearly visible size
                    color='red',     # Make the point stand out
                    symbol='circle'
                ),
                name=f"Point {tuple(coordinates)}"
            )

            # --- 7. Create figure and layout ---
            fig = go.Figure(data=[mesh_trace, point_trace])

            fig.update_layout(
                title=f"Mesh and Point for {filename_key}",
                scene=dict(
                    xaxis_title='X Axis',
                    yaxis_title='Y Axis',
                    zaxis_title='Z Axis',
                    # This ensures the plot is not distorted
                    aspectmode='data' 
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )

            # --- 8. Save the interactive HTML file ---
            output_html_path = output_dir / f"{filename_key}.html"
            fig.write_html(str(output_html_path))
            processed_count += 1

        except Exception as e:
            print(f"Error processing {filename_key} from {stl_path}: {e}. Skipping.")
            skipped_count += 1

    # --- 9. Final summary ---
    print("\n--- Processing Complete ---")
    print(f"Successfully created {processed_count} plots.")
    print(f"Skipped {skipped_count} entries (due to missing files or errors).")

if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Create interactive 3D plots from JSON coordinates and STL files."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing coordinates."
    )
    parser.add_argument(
        "--stl_dirs",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to the directory/directories containing the STL files. You can specify multiple directories separated by spaces."
    )

    args = parser.parse_args()
    
    # Run the main function
    create_plots(args.json_file, args.stl_dirs)