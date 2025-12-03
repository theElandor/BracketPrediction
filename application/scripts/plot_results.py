import os
import json
import re
from pathlib import Path
from stl import mesh
import numpy as np
import plotly.graph_objects as go

# --- plotting function ---
def plot_stl_with_points_interactive(stl_path: str, bracket, out_path_png: Path, out_path_html: Path):
    your_mesh = mesh.Mesh.from_file(stl_path)
    vertices = your_mesh.vectors.reshape(-1, 3)
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = np.arange(0, len(vertices), 3), np.arange(1, len(vertices), 3), np.arange(2, len(vertices), 3)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='lightgray',
        opacity=0.5,
        flatshading=True,
        name='Mesh'
    ))

    p1 = np.array(bracket, dtype=float)
    fig.add_trace(go.Scatter3d(
        x=[p1[0]], y=[p1[1]], z=[p1[2]],
        mode='markers+text',
        text=["Pred"],
        textposition="top center",
        marker=dict(size=6, color='red'),
        name='Prediction'
    ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        width=900, height=700,
        title=os.path.basename(stl_path)
    )

    # Save both static and interactive versions
    fig.write_image(out_path_png)
    fig.write_html(out_path_html, include_plotlyjs='cdn')


# --- directories ---
data_dir = Path('/work/grana_maxillo/Mlugli/brackets_melted/flattened')
predictions_dir = Path('/work/grana_maxillo/Mlugli/brackets_melted/model_predictions')
json_dir = predictions_dir / 'json'
plot_dir = predictions_dir / 'plot'
plot_dir.mkdir(parents=True, exist_ok=True)

# --- find latest epoch per sample ---
pattern = re.compile(r'^(?P<name>.+)_epoch(?P<epoch>\d+)\.json$')
latest_files = {}

for json_file in json_dir.glob("*.json"):
    match = pattern.match(json_file.name)
    if not match:
        continue
    name, epoch = match.group("name"), int(match.group("epoch"))
    if name not in latest_files or epoch > latest_files[name][0]:
        latest_files[name] = (epoch, json_file)

# --- plot predictions for latest epoch ---
for name, (epoch, json_file) in latest_files.items():
    with open(json_file, "r") as f:
        data = json.load(f)
    coords = data["coords"]

    stl_path = data_dir / f"{name}.stl"
    if not stl_path.exists():
        print(f"⚠️ STL not found for {name}, skipping.")
        continue

    out_path_png = plot_dir / f"{name}.png"
    out_path_html = plot_dir / f"{name}.html"

    print(f"Plotting {name} (epoch {epoch}) → {out_path_png.name} & {out_path_html.name}")
    plot_stl_with_points_interactive(str(stl_path), coords, out_path_png, out_path_html)
