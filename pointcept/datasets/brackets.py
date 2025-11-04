"""
Brackets dataset

Author: Matteo Lugli (283122@studenti.unimore.it)
Please cite our work if the code is helpful to you.
"""

import os
import json
import numpy as np
from torch.utils.data import Dataset
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose
from pointcept.datasets.defaults import DefaultDataset
from pathlib import Path
import trimesh
import plotly.graph_objects as go
from stl import mesh
import random
from copy import deepcopy
from pointcept.datasets.transform import TRANSFORMS  

@DATASETS.register_module()
class BracketPointDataset(DefaultDataset):
    """
    Dataset for predicting bracket_point from STL files.
    """
 
    def __init__(
        self,
        data_root,
        point_count:int = 8192,
        split="train",
        fold = 1,
        transform=None,
        test_mode=False,
        test_cfg=None,
        plot=False,
        loop=1,
    ):
        self.fold = fold
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )
        self.point_count = point_count
        self.plot = plot
        if test_mode:
            self.post_transform = Compose(test_cfg.post_transform)  
            self.aug_transform = [Compose(aug) for aug in test_cfg.aug_transform]

    def plot_stl_with_points_interactive(self, stl_path, bracket):
        # Load STL mesh
        debug_plots_path = "/work/grana_maxillo/Mlugli/debug_plots"
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
 
    def get_data_list(self):
        """Load list of data samples from fold JSON files."""
        # Load the appropriate fold file
        fold_file = os.path.join(self.data_root, f"split_{self.fold}.json")
        
        if not os.path.exists(fold_file):
            raise FileNotFoundError(
                f"Split file not found: {fold_file}\n"
                f"Please run the split generation script first to create split_{self.fold}.json"
            )
        
        with open(fold_file, 'r') as f:
            split_data = json.load(f)
        
        # Map split names
        split_mapping = {
            'train': 'train',
            'val': 'validation',
            'test': 'test'
        }
        
        split_key = split_mapping.get(self.split)
        if split_key not in split_data:
            raise ValueError(f"Invalid split: {self.split}. Must be one of {list(split_mapping.keys())}")
        
        # Get file paths from the split
        file_paths = split_data[split_key]['files']
        
        # Filter only .stl files and extract file names without extension
        # Also remove the path prefix to get just the filename
        file_names = []
        for file_path in file_paths:
            if file_path.endswith('.stl'):
                # Extract just the filename without extension
                # e.g., "Brackets/brackets_1_melted/flattened/STEM_lower_1_FDI_31.stl" 
                # -> "STEM_lower_1_FDI_31"
                file_names.append(file_path)
                #file_name = os.path.basename(file_path).replace('.stl', '')
                #file_names.append(file_name)
        
        print(f"Loaded {len(file_names)} samples from fold {self.fold}, split {self.split}")
        print(f"  Patients: {len(split_data[split_key]['patient_ids'])}")
        print(f"  Total files (stl+json): {split_data[split_key]['num_files']}")
        
        return file_names
    
    def _load_stl(self, stl_path):
        mesh = trimesh.load(stl_path, force='mesh')
        points, normals = trimesh.sample.sample_surface(mesh, count=self.point_count)
        return points.astype(np.float32)
    
    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        bracket_point = np.array(data['bracket'], dtype=np.float32)
        try:
            facial_point = np.array(data['facial'], dtype=np.float32)
        except:
            facial_point = None
        return bracket_point, facial_point
    
    def get_data(self, idx, testing=False):  
        file_rel_path = self.data_list[idx % len(self.data_list)]  
        stl_path = os.path.join(self.data_root, file_rel_path)  
        json_path = os.path.join(self.data_root, file_rel_path.replace(".stl", ".json"))  
        
        coord = self._load_stl(stl_path)
        bracket_point, facial_point = self._load_json(json_path)
        d = {  
            "coord": coord,  
            "name": Path(stl_path).stem,
            "bracket": bracket_point
        }
        # some corrections
        if facial_point is not None:
            d["facial"] = facial_point
        if testing:
            d["segment"] = bracket_point
        return d
    
    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx, testing=True)  
        data_dict = self.transform(data_dict)  
        
        # Extract ground truth bracket_point and segment  
        result_dict = dict(  
            segment=data_dict.pop("segment"),  
            bracket_point=data_dict.pop("bracket"),  # Add this line 
            facial_point = data_dict.pop("facial"), 
            name=data_dict.pop("name")  
        )  
        
        if "origin_segment" in data_dict:  
            assert "inverse" in data_dict  
            result_dict["origin_segment"] = data_dict.pop("origin_segment")  
            result_dict["inverse"] = data_dict.pop("inverse")  
    
        # Create fragments with augmentations
        data_dict_list = []  
        for aug in self.aug_transform:  
            data_dict_list.append(aug(deepcopy(data_dict)))  
    
        fragment_list = []  
        for data in data_dict_list:  
            if self.test_voxelize is not None:  
                data_part_list = self.test_voxelize(data)  
            else:  
                data["index"] = np.arange(data["coord"].shape[0])  
                data_part_list = [data]  
            for data_part in data_part_list:  
                if self.test_crop is not None:  
                    data_part = self.test_crop(data_part)  
                else:  
                    data_part = [data_part]  
                fragment_list += data_part  
    
        for i in range(len(fragment_list)):  
            fragment_list[i] = self.post_transform(fragment_list[i])  
        result_dict["fragment_list"] = fragment_list  
        return result_dict
    
    def __len__(self):
        return len(self.data_list) * self.loop