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
class BracketPointDataset(DefaultDataset):
    """
    Dataset for predicting bracket_point from STL files.
    """
 
    def __init__(
        self,
        data_root,
        point_count:int = 8192,
        split="train",
        debug=False,
        transform=None,
        test_mode=False,
        test_cfg=None,
        plot=False,
        loop=1,
        fold = 1, # fold to use
        debased=False, # use debased meshes
    ):
        self.fold = fold
        self.debased = debased
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
        self.debug = debug
        self.fallbacks = 0 # counter for malformed stl files
        if test_mode:
            self.post_transform = Compose(test_cfg.post_transform)  
            self.aug_transform = [Compose(aug) for aug in test_cfg.aug_transform]
 
    def get_data_list(self):
        """Load list of data samples from fold JSON files."""
        fold_file = os.path.join(self.data_root, f"split_{self.fold}.json")
        if not os.path.exists(fold_file):
            raise FileNotFoundError(
                f"Split file not found: {fold_file}\n"
                f"Please run the split generation script first to create split_{self.fold}.json"
            )
        with open(fold_file, 'r') as f:
            split_data = json.load(f)
        split_mapping = {
            'train': 'train',
            'val': 'validation',
            'test': 'test'
        }
        split_key = split_mapping.get(self.split)
        if split_key not in split_data:
            raise ValueError(f"Invalid split: {self.split}. Must be one of {list(split_mapping.keys())}")

        # Get file paths from the split
        if not self.debased:
            file_paths = split_data[split_key]['files']
        else: # for samples in brackets_2 change path and take the debased file.
            file_paths = [
                x.replace('flattened', 'flattened_debased') if 'brackets_2' in x else x
                for x in split_data[split_key]['files']
            ]
        
        # Filter only .stl files and extract file names without extension
        # Also remove the path prefix to get just the filename
        file_names = []
        for file_path in file_paths:
            if file_path.endswith('.stl'):
                file_names.append(file_path)
        print(f"Loaded {len(file_names)} samples from fold {self.fold}, split {self.split}")
        print(f"  Patients: {len(split_data[split_key]['patient_ids'])}")
        print(f"  Total files (stl+json): {split_data[split_key]['num_files']}")
        
        return file_names
    
    def _load_stl(self, stl_path):
        try:
            # Try loading the primary file
            mesh = trimesh.load(stl_path, force='mesh')

        except Exception as e_primary:
            # If primary fails, log it and try the fallback
            self.fallbacks += 1
            based_file = stl_path.replace("flattened_debased", "flattened")
            print(f"'{stl_path}' is broken (Error: {e_primary}). Fallback to '{based_file}'")
            
            try:
                mesh = trimesh.load(based_file, force='mesh')
            except Exception as e_fallback:
                # If fallback also fails, log it and return None
                print(f"FATAL: Fallback '{based_file}' also failed (Error: {e_fallback}).")
                return None, None # Cannot proceed

        # If we get here, 'mesh' is loaded. Now we can sample
        try:
            # 1. Sample the surface to get points and the face index for each point
            points, face_index = trimesh.sample.sample_surface(mesh, count=self.point_count)

            # 2. Get the normals for those faces
            normals = mesh.face_normals[face_index]

            return points.astype(np.float32), normals.astype(np.float32)

        except Exception as e_sample:
            # Handle cases where the mesh loaded but couldn't be sampled
            print(f"Loaded mesh '{mesh.metadata.get('file_name')}' but failed to sample: {e_sample}")
            return None, None
   
    
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
        
        coord, normal = self._load_stl(stl_path)
        bracket_point, facial_point = self._load_json(json_path)
        assert type(coord) != type(None), f"Sample {stl_path} coordinates not loaded correctly."
        assert type(normal) != type(None), f"Could not load normal for sample {stl_path}."

        assert coord.shape == (self.point_count, 3), f"Wrongly shaped coordinates for sample {stl_path}"
        assert normal.shape == (self.point_count, 3), f"Wrongly shaped normals for sample {stl_path}"
        assert bracket_point.shape == (3,), f"Wrongly shaped bracket point for sample {stl_path}"
        d = {
            "coord": coord,  
            "normal": normal,
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
            bracket =data_dict.pop("bracket"),  # Add this line 
            name=data_dict.pop("name")
        )  
        
        if "facial" in data_dict:
            result_dict["facial"] = data_dict.pop("facial")

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