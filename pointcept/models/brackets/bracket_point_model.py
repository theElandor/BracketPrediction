"""
Model for Bracket Point Prediction
Place this file in: pointcept/models/bracket_point_model.py
"""
import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from torch_scatter import segment_csr  

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
    """  
      
    def __init__(  
        self,
        backbone,  
        backbone_out_channels=96,  
        output_dim=3,  # 3D point coordinates
        save_predictions=False,
        output_dir:str = "output"
    ):  
        super().__init__()
          
        self.backbone = build_model(backbone)  
        self.output_dir = output_dir
        self.save_predictions = save_predictions

        # Regression head: outputs 3D point coordinates  
        self.head = nn.Sequential(  
            nn.Linear(backbone_out_channels, 256),  
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
     
        # Predict 3D point    
        bracket_point_pred = self.head(feat)    
        # during training we just need to store the loss value in the output dict,
        # while during testing we also need to store the prediction itself so that the
        # tester can access the field.
        if not self.training:
            out = {"bracket_point_pred":bracket_point_pred} # Add predictions to output
        else:
            out = {}

        # Compute MSE loss if ground truth available    
        if "bracket_point" in input_dict:
            # here the "target" tensor is of shape [B*3], while the
            # "bracket_point_pred" tensor is of shape [B,3]". This
            # causes an error in nn.functional.mse_loss.
            # So we need to reshape the target tensor. (I believe is what we need to do)
            target = input_dict["bracket_point"].view_as(bracket_point_pred)
            if self.save_predictions and not self.training: # plot only in inference.
                self._save(input_dict, bracket_point_pred)
            loss = nn.functional.mse_loss(bracket_point_pred, target)
            out["loss"] = loss

        return out