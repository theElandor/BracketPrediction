
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
        transform=None,
        test_mode=False,
        test_cfg=None,
        plot=False,
        loop=1,
    ):
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
        #self.data_list = self._load_data_list()
        if test_mode:
            self.post_transform = Compose(test_cfg.post_transform)  
            self.aug_transform = [Compose(aug) for aug in test_cfg.aug_transform]

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
 
    def get_data_list(self):
        """Load list of data samples."""
        split_file = os.path.join(self.data_root, f"{self.split}.txt")
        if os.path.exists(split_file):
            # If split file exists, use it
            with open(split_file, 'r') as f:
                file_names = [line.strip() for line in f.readlines()]
        else:
            file_names = []
            file_names = [f.replace('.stl', '') for f in os.listdir(self.data_root) if f.endswith('.stl')]
            # temporary, but for initial consistency it is fine.
            random.seed(42)
            random.shuffle(file_names)
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
    
    def get_data(self, idx):  
        file_name = self.data_list[idx % len(self.data_list)]  
        stl_path = os.path.join(self.data_root, f"{file_name}.stl")  
        json_path = os.path.join(self.data_root, f"{file_name}.json")  
        
        coord = self._load_stl(stl_path)  
        bracket_point = self._load_json(json_path)  
        
        return {  
            "coord": coord,  
            "name": file_name,  
            "bracket_point": bracket_point,  
            #"segment": bracket_point,  # For compatibility  
        }
    
    def prepare_test_data(self, idx):  
        # Load data  
        data_dict = self.get_data(idx)  
        data_dict = self.transform(data_dict)  
        
        # Extract ground truth bracket_point and segment  
        result_dict = dict(  
            segment=data_dict.pop("segment"),  
            bracket_point=data_dict.pop("bracket_point"),  # Add this line  
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
    #def __getitem__(self, idx):
    #    # Handle looping
    #    idx = idx % len(self.data_list)
    #    file_name = self.data_list[idx]
    #
    #    # Load STL and JSON
    #    stl_path = os.path.join(self.data_root, f"{file_name}.stl")
    #    json_path = os.path.join(self.data_root, f"{file_name}.json")
    #    
    #    # Load point cloud from STL
    #    coord = self._load_stl(stl_path)
    #    
    #    # Load target bracket_point
    #    bracket_point = self._load_json(json_path)
    #    
    #    if self.plot:
    #        self.plot_stl_with_points_interactive(stl_path, bracket_point)
    #    # Create data dict
    #    data_dict = {
    #        "coord": coord,
    #        "name": file_name,
    #        "bracket_point": bracket_point,
    #    }
    #    
    #    # Apply transforms
    #    if self.transform is not None:
    #        data_dict = self.transform(data_dict)
    #    if self.test_mode:  
    #        fragment_list = []  
    #        data_dict_list = []  
    #        
    #        # Apply test voxelization  
    #        data_dict = self.test_voxelize(data_dict)  
    #        
    #        # Apply test crop if configured  
    #        if self.test_crop:  
    #            data_dict_list = self.test_crop(data_dict)  
    #        else:  
    #            data_dict_list = [data_dict]  
    #        
    #        # Apply augmentations and post-transforms  
    #        for data_dict in data_dict_list:  
    #            for aug in self.aug_transform:  
    #                data_dict_aug = aug(deepcopy(data_dict))  
    #                fragment_list.append(self.post_transform(data_dict_aug))  
    #        
    #        return fragment_list
    #    else:  
    #        return data_dict
    
    def __len__(self):
        return len(self.data_list) * self.loop