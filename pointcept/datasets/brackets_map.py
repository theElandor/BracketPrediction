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
from copy import deepcopy
from pointcept.datasets.transform import TRANSFORMS

@DATASETS.register_module()
class BracketMapDataset(DefaultDataset):
    """
    Dataset for predicting bracket_point from STL files.
    """
 
    def __init__(
        self,
        data_root,
        split="train",
        debug=False,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        fold = 1, # fold to use
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
        self.debug = debug
        self.fallbacks = 0 # counter for malformed stl files
        if test_mode:
            self.post_transform = Compose(test_cfg.post_transform)  
            self.aug_transform = [Compose(aug) for aug in test_cfg.aug_transform]
 
    def get_data_list(self):
        """Load list of data samples from data_root with automatic 80/10/10 split."""
        # Find all .stl files in data_root
        all_stl_files = []
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith('.stl'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.data_root)
                    # Check if corresponding *_softlabel.npy file exists
                    npy_path = os.path.join(self.data_root, rel_path.replace('.stl', '_softlabel.npy'))
                    if os.path.exists(npy_path):
                        all_stl_files.append(rel_path)
        
        if not all_stl_files:
            raise ValueError(f"No .stl files with corresponding *_softlabel.npy files found in {self.data_root}")
        
        # Create deterministic 80/10/10 split using fixed seed
        np.random.seed(42)
        indices = np.arange(len(all_stl_files))
        np.random.shuffle(indices)
        
        num_train = int(0.8 * len(all_stl_files))
        num_val = int(0.1 * len(all_stl_files))
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]
        
        split_mapping = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }

        if self.split not in split_mapping:
            raise ValueError(f"Invalid split: {self.split}. Must be one of {list(split_mapping.keys())}")
        
        selected_indices = split_mapping[self.split]
        file_names = [all_stl_files[i] for i in selected_indices]
        
        print(f"Loaded {len(file_names)} samples from split {self.split}")
        print(f"  Total train: {len(split_mapping['train'])}, val: {len(split_mapping['val'])}, test: {len(split_mapping['test'])}")
        
        return file_names
    
    def _load_stl(self, stl_path):
        try:
            # Try loading the primary file
            mesh = trimesh.load(stl_path, force='mesh')
            points = mesh.vertices
            normals = mesh.vertex_normals
            return points.astype(np.float32), normals.astype(np.float32)
        except:
            print(f"Couldn't load sample {stl_path}")
            raise
   
    def _load_heatmap(self, npy_path):  
        """Load heatmap values from .npy file"""  
        try:  
            heatmap = np.load(npy_path)  
            # Ensure it's float32 and 1D array  
            heatmap = heatmap.astype(np.float32).reshape(-1)  
            return heatmap  
        except Exception as e:  
            print(f"Couldn't load heatmap {npy_path}: {e}")  
            raise
    
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
        heatmap_path = os.path.join(self.data_root, file_rel_path.replace(".stl", "_softlabel.npy"))
        
        coord, normal = self._load_stl(stl_path)
        bracket_point, facial_point = self._load_json(json_path)
        segment = self._load_heatmap(heatmap_path)
        assert segment.shape[0] == coord.shape[0], f"Segment shape not matching for sample {stl_path}"
        d = {
            "coord": coord,  
            "normal": normal,
            "name": Path(stl_path).stem,
            "bracket": bracket_point,
            "segment": segment # heatmap to predict
        }
        # some corrections
        if facial_point is not None:
            d["facial"] = facial_point
        return d
    
    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx, testing=True)  
        data_dict = self.transform(data_dict)
        
        # ============Extract ground truth data==============
        result_dict = dict(
            segment=data_dict.pop("segment"),  
            bracket=data_dict.pop("bracket"),
            name=data_dict.get("name")
        ) 
        if "facial" in data_dict:
            result_dict["facial"] = data_dict.pop("facial")
        # ==================================================

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
        if self.debug: return 2 # if debugging, run on just 2 samples
        return len(self.data_list) * self.loop