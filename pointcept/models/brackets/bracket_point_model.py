"""
Model for Bracket Point Prediction
Place this file in: pointcept/models/bracket_point_model.py
"""
import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from torch_scatter import segment_csr  
import torch.nn.functional as F
import wandb

from ..builder import build_model  
from pointcept.models.utils.structure import Point  
import numpy as np
import plotly.graph_objects as go
from stl import mesh
import json
import os
from collections import defaultdict

@MODELS.register_module()  
class VoxelBracketPredictor(nn.Module):  
    """  
    - Voxel-based backbone (PTv3, SPUnet)
    - Simple regression head
    - Modes:
        1) Normal mode:
            Directly regress the bracket point
        2) Offset mode:
            Predicts the offset from the "facial" landmark
            extracted from 3D teethland. Requires a "facial" point.
            field with 3 coordinates in the json files of the
            dataset.
    """  
      
    def __init__(  
        self,
        backbone,
        mode="normal",
        alpha = 0.0,
        backbone_out_channels=96,
        output_dim=3,  # 3D point coordinates
        save_predictions=False,
        class_embedding=False,
        output_dir:str = "output"
    ):  
        super().__init__()
          
        self.backbone = build_model(backbone)
        self.alpha = alpha 
        self.mode = mode
        if self.mode not in "offset normal".split():
            raise ValueError("Unknown model mode. Please use either normal or offset.")
        self.output_dir = output_dir
        self.save_predictions = save_predictions
        self.class_embedding=class_embedding
        self.num_classes = 8 # FDI index % 10
        self.tooth_embedding_dim = 128

        # embedding layer for the one hot tooth class
        self.embedder = nn.Embedding(self.num_classes, self.tooth_embedding_dim)

        # Regression head: outputs 3D point coordinates  
        head_input = backbone_out_channels if not self.class_embedding else backbone_out_channels + self.tooth_embedding_dim
        self.head = nn.Sequential( 
            nn.Linear(head_input, 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),
        )
        
    def _save(self, input_dict: dict, bracket_point_pred: torch.Tensor) -> None:
        """
        Writes the predicted points as JSON files.

        Each file will be named:
            {sample_name}_epoch{N}.json
        where N is how many times this function has been called for that sample.

        Args:
            input_dict (dict): Contains "name" (list of sample names without extension).
            bracket_point_pred (torch.Tensor): Predictions of shape [B, 3].
        """
        # Initialize a counter the first time this is called
        if not hasattr(self, "_save_counter"):
            self._save_counter = defaultdict(int)

        os.makedirs(self.output_dir, exist_ok=True)

        names = input_dict["name"]
        preds = bracket_point_pred.detach().cpu().numpy()  # convert to numpy for JSON serialization

        for name, coords in zip(names, preds):
            # Increment epoch counter for this specific sample
            self._save_counter[name] += 1
            epoch_idx = self._save_counter[name]

            # Construct output filename
            filename = f"{name}_epoch{epoch_idx}.json"
            filepath = os.path.join(self.output_dir, filename)

            # Write coordinates to JSON
            with open(filepath, "w") as f:
                json.dump({"coords": coords.tolist()}, f, indent=4)


    def forward(self, input_dict):
        point = self.backbone(input_dict)
        # Handle Point structure from voxel-based backbones    
        if isinstance(point, Point):    
            # Global average pooling across all points    
            point.feat = segment_csr(    
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),    
                reduce="mean",
            )    
            feat = point.feat    
        else:
            feat = point
        if self.class_embedding:
            class_indices = torch.tensor([(int(x.split("_")[-1]) % 10)-1 for x in input_dict["name"]], device=feat.device).long()
            # Check validity
            invalid_mask = (class_indices < 0) | (class_indices > 8)
            if invalid_mask.any(): print("Warning: clamping invalid class indices:", class_indices[invalid_mask].tolist())
            class_indices = torch.clamp(class_indices, min=0, max=7)

            class_embeddings = self.embedder(class_indices)
            input_tensor = torch.cat([feat, class_embeddings], dim=1)
        else:
            input_tensor = feat 

        # Regress position of 3D points.

        prediction= self.head(input_tensor)
        
        if self.mode == "normal":
            bracket_point_pred = prediction
        else:
            bracket_point_pred = input_dict["facial"].view_as(prediction) + prediction
        if not self.training:
            out = {"bracket_point_pred":bracket_point_pred} # Add predictions to output
        else:
            out = {}
        # Compute MSE loss if ground truth available
        if "bracket" in input_dict:
            # here the "target" tensor is of shape [B*3], while the
            # "bracket_point_pred" tensor is of shape [B,3]". This
            # causes an error in nn.functional.mse_loss.
            # So we need to reshape the target tensor (I guess).
            target = input_dict["bracket"].view_as(bracket_point_pred)
            # if in normal mode, bracket_point_pred is the final prediction
            # if in offset mode, bracket_point_pred is the offset to add
            # to toothInstanceNet "facial_point" landmark. 
            if self.save_predictions and not self.training: # plot only in inference.
                self._save(input_dict, bracket_point_pred)
            mse = nn.functional.mse_loss(bracket_point_pred, target)
            cos_sim = nn.functional.cosine_similarity(bracket_point_pred, target).mean()
            out["cos_sim"] = cos_sim.detach() if self.alpha == 0 else cos_sim
            out["mse"] = mse
            if self.alpha != 0:
                out["loss"] = (1 - self.alpha) * mse + self.alpha * (1-cos_sim) # MSE and cosine distance
            else:
                out["loss"] = mse
        return out


@MODELS.register_module()    
class VoxelBracketPredictor_v2(nn.Module):    
    """    
    Two-stage bracket point prediction model:  
    1. Coarse prediction head (class-agnostic)  
    2. Per-class refinement heads that predict offsets  
      
    Final prediction = coarse_pred + class_specific_offset  
    """    
        
    def __init__(    
        self,
        backbone,  
        alpha=0.0,  
        backbone_out_channels=96,  
        output_dim=3,  
        save_predictions=False,  
        output_dir: str = "output",
        use_refinement = False,
        freeze_backbone = False,
        freeze_coarse = False,  
    ): 
        super().__init__()  
            
        self.backbone = build_model(backbone)  
        self.alpha = alpha   
        self.output_dir = output_dir  
        self.save_predictions = save_predictions  
        self.num_classes = 4  # Reduced from 8 to 4 grouped classess
        self.use_refinement = use_refinement
        self.freeze_backbone = freeze_backbone
        self.freeze_coarse = freeze_coarse
        
        # Mapping from original tooth indices (0-7) to grouped class indices (0-3)  
        self.class_mapping = {  
            0: 0,  # indices 0,1 -> group 0  
            1: 0,  
            2: 1,  # index 2 -> group 1  
            3: 2,  # indices 3,4 -> group 2  
            4: 2,  
            5: 3,  # indices 5,6 -> group 3  
            6: 3,  
            7: 3,  # index 7 also maps to group 3 (assuming you want this)  
        }  
    
        # Coarse prediction head (class-agnostic)  
        self.coarse_head = nn.Sequential(  
            nn.Linear(backbone_out_channels, 256),  
            nn.LayerNorm(256),  
            nn.ReLU(inplace=True),  
            nn.Dropout(p=0.3),  
            nn.Linear(256, 128),  
            nn.LayerNorm(128),  
            nn.ReLU(inplace=True),  
            nn.Dropout(p=0.3),  
            nn.Linear(128, output_dim),  
        )  
        
        # Per-group refinement heads (now only 4 instead of 8)  
        self.refinement_heads = nn.ModuleList([  
            nn.Sequential(  
                nn.Linear(backbone_out_channels, 128),  
                nn.LayerNorm(128),  
                nn.ReLU(inplace=True),  
                nn.Dropout(p=0.3),  
                nn.Linear(128, 64),  
                nn.LayerNorm(64),  
                nn.ReLU(inplace=True),  
                nn.Dropout(p=0.3),  
                nn.Linear(64, output_dim),  
            )  
            for _ in range(self.num_classes)  
        ])
 
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        if self.freeze_coarse:
            for param in self.coarse_head.parameters():
                param.requires_grad = False
          
    def _save(self, input_dict: dict, bracket_point_pred: torch.Tensor) -> None:  
        """  
        Writes the predicted points as JSON files.  
  
        Each file will be named:
            {sample_name}_epoch{N}.json
        where N is how many times this function has been called for that sample.
  
        Args:  
            input_dict (dict): Contains "name" (list of sample names without extension).  
            bracket_point_pred (torch.Tensor): Predictions of shape [B, 3].  
        """  
        # Initialize a counter the first time this is called  
        if not hasattr(self, "_save_counter"):  
            self._save_counter = defaultdict(int)  
  
        os.makedirs(self.output_dir, exist_ok=True)  
  
        names = input_dict["name"]  
        preds = bracket_point_pred.detach().cpu().numpy()  # convert to numpy for JSON serialization  
  
        for name, coords in zip(names, preds):  
            # Increment epoch counter for this specific sample  
            self._save_counter[name] += 1  
            epoch_idx = self._save_counter[name]  
  
            # Construct output filename  
            filename = f"{name}_epoch{epoch_idx}.json"  
            filepath = os.path.join(self.output_dir, filename)  
  
            # Write coordinates to JSON  
            with open(filepath, "w") as f:  
                json.dump({"coords": coords.tolist()}, f, indent=4)  
  
    def forward(self, input_dict):  
        point = self.backbone(input_dict)  
        
        if isinstance(point, Point):      
            point.feat = segment_csr(      
                src=point.feat,  
                indptr=nn.functional.pad(point.offset, (1, 0)),      
                reduce="mean",
            )      
            feat = point.feat
        else:  
            feat = point  
        
        # Coarse prediction (class-agnostic)  
        coarse_pred = self.coarse_head(feat)  
        if self.use_refinement: 
            # Extract original class indices from sample names  
            # This check is bad, will be removed in the future.
            if type(input_dict["name"]) == type(""): # then we are in test mode
                original_class_indices = torch.tensor([(int(input_dict["name"].split("_")[-1]) % 10) - 1]).long()
            else:
                original_class_indices = torch.tensor(  
                    [(int(x.split("_")[-1]) % 10) - 1 for x in input_dict["name"]],  
                    device=feat.device  
                ).long()
            original_class_indices = torch.clamp(original_class_indices, min=0, max=7) 
            
            # Map to grouped class indices  
            grouped_class_indices = torch.tensor(  
                [self.class_mapping[idx.item()] for idx in original_class_indices],  
                device=feat.device  
            ).long()
            
            # Get class-specific refinement offsets using grouped indices  
            refinement_offsets = torch.zeros_like(coarse_pred)  
            for i in range(self.num_classes):  
                mask = (grouped_class_indices == i)  
                if mask.any():  
                    refinement_offsets[mask] = self.refinement_heads[i](feat[mask]) 
            
            # Final prediction: coarse + class-specific offset  
            bracket_point_pred = coarse_pred + refinement_offsets
        else: # use only class agnostic prediction otherwise.
            bracket_point_pred = coarse_pred

        # Compute ground truth offset that has to be predicted: 
        if not self.training:
            out = {"bracket_point_pred": bracket_point_pred}  # Add predictions to output  
        else:
            out = {}
          
        # Compute MSE loss if ground truth available
        if "bracket" in input_dict:
            target = input_dict["bracket"].view_as(bracket_point_pred) 
            if self.save_predictions and not self.training:  # plot only in inference.  
                self._save(input_dict, bracket_point_pred)  

            mse = nn.functional.mse_loss(bracket_point_pred, target)  
            cos_sim = nn.functional.cosine_similarity(bracket_point_pred, target).mean()  
            out["cos_sim"] = cos_sim.detach() if self.alpha == 0 else cos_sim  
            out["mse"] = mse  
            if self.use_refinement:
                # in this case the error is between the target offset and the predicted offset
                target_offset = target - coarse_pred
                #mse = nn.functional.mse_loss(bracket_point_pred - coarse_pred, target_offset)
                huber = nn.functional.huber_loss(bracket_point_pred-coarse_pred, target_offset, delta=1e-2)
                out["loss"] = huber
            else:
                if self.alpha != 0:  
                    out["loss"] = (1 - self.alpha) * mse + self.alpha * (1 - cos_sim)  # MSE and cosine distance  
                else:  
                    out["loss"] = mse  
          
        return out