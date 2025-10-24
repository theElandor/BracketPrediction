import os
import json
import numpy as np
from torch.utils.data import Dataset
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose
from pathlib import Path
import trimesh
import plotly.graph_objects as go
from stl import mesh


@DATASETS.register_module()
class BracketPointDataset(Dataset):
    """
    Dataset for predicting bracket_point from STL files.
    """
 
    def __init__(
        self,
        data_root,
        point_count:int = 8192,
        split="train",
        transform=None,
        test_mode=False,
        plot=False,
        loop=1,
    ):
        super().__init__()
        self.data_root = data_root
        self.point_count = point_count
        self.split = split
        self.transform = Compose(transform) if transform is not None else None
        self.test_mode = test_mode
        self.loop = loop
        self.plot = plot
        self.data_list = self._load_data_list()

    def plot_stl_with_points_interactive(self, stl_path, bracket):
        # Load STL mesh
        # STL path is like /work/grana_maxillo/Mlugli/brackets_melted/stl/FDI_22/STEM_upper_33_FDI_22.stl
        debug_plots_path = "/work/grana_maxillo/Mlugli/brackets_melted/debug_plots"
        your_mesh = mesh.Mesh.from_file(stl_path)
        vertices = your_mesh.vectors.reshape(-1, 3)
        
        # Create triangles for Plotly
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = (
            np.arange(0, len(vertices), 3),
            np.arange(1, len(vertices), 3),
            np.arange(2, len(vertices), 3),
        )

        # Create Plotly figure
        fig = go.Figure()

        # Add mesh
        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color='lightgray',
                opacity=0.5,
                flatshading=True,
                name='Mesh'
            )
        )

        # Add points
        p1 = np.array(bracket, dtype=float)
        fig.add_trace(go.Scatter3d(
            x=[p1[0]], y=[p1[1]], z=[p1[2]],
            mode='markers+text',
            text=["P1"],
            textposition="top center",
            marker=dict(size=6, color='red'),
            name='Gt'
        ))
        
        # Layout and camera
        fig.update_layout(
            title=f"Interactive STL Visualization ({stl_path})",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=True,
            width=900,
            height=700,
        )
        full_path = Path(stl_path)
        base_name = full_path.stem
        out_filename = Path(debug_plots_path) / f"{base_name}.png"
        fig.write_image(out_filename)
 
    def _load_data_list(self):
        """Load list of data samples."""
        split_file = os.path.join(self.data_root, f"{self.split}.txt")
        if os.path.exists(split_file):
            # If split file exists, use it
            with open(split_file, 'r') as f:
                file_names = [line.strip() for line in f.readlines()]
        else:
            file_names = []
            file_names = [f.replace('.stl', '') for f in os.listdir(self.data_root) if f.endswith('.stl')]
            if self.split == 'train':
                file_names = file_names[:int(len(file_names) * 0.8)]
            elif self.split == 'val':
                file_names = file_names[int(len(file_names) * 0.8):int(len(file_names) * 0.9)]
            elif self.split == 'test':
                file_names = file_names[int(len(file_names) * 0.9):]
        
        return file_names
    
    def _load_stl(self, stl_path):
        mesh = trimesh.load(stl_path, force='mesh')
        points, normals = trimesh.sample.sample_surface(mesh, count=self.point_count)

        # did this cuz I did not know how pointcept batching logic worked.
        #points = np.expand_dims(points.astype(np.float32), axis=0)
        return points.astype(np.float32)
    
    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        bracket_point = np.array(data['bracket_point'], dtype=np.float32)
        #bracket_point = np.expand_dims(bracket_point, axis=0)
        return bracket_point
    
    def __getitem__(self, idx):
        # Handle looping
        idx = idx % len(self.data_list)
        file_name = self.data_list[idx]

        # Load STL and JSON
        stl_path = os.path.join(self.data_root, f"{file_name}.stl")
        json_path = os.path.join(self.data_root, f"{file_name}.json")
        
        # Load point cloud from STL
        coord = self._load_stl(stl_path)
        
        # Load target bracket_point
        bracket_point = self._load_json(json_path)
        
        if self.plot:
            self.plot_stl_with_points_interactive(stl_path, bracket_point)
        # Create data dict
        data_dict = {
            "coord": coord,
            "name": file_name,
            "bracket_point": bracket_point,
        }
        
        # Apply transforms
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        
        return data_dict
    
    def __len__(self):
        return len(self.data_list) * self.loop