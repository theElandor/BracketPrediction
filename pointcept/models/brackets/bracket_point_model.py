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

from ..builder import build_model  
from pointcept.models.utils.structure import Point  
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
        class_embedding=False,
        output_dir:str = "output",
        mae_weight = 1.0,  # Weight for MSE loss
    ):  
        super().__init__()  
          
        self.backbone = build_model(backbone)
        self.output_dir = output_dir
        self.class_embedding=class_embedding

        # ========= loss weights =============
        self.mae_weight = mae_weight

        self.num_classes = 8 # FDI index % 10
        self.tooth_embedding_dim = 128
        # embedding layer for the one hot tooth class
        self.embedder = nn.Embedding(self.num_classes, self.tooth_embedding_dim)
        # Separate regression heads for each point (bracket, incisal, outer)
        head_input = backbone_out_channels if not self.class_embedding else backbone_out_channels + self.tooth_embedding_dim
        
        # Bracket head
        self.bracket_head = nn.Sequential( 
            nn.Linear(head_input, 256),  
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),  # 3 coordinates
        )
        
        # Incisal head
        self.incisal_head = nn.Sequential( 
            nn.Linear(head_input, 256),  
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),  # 3 coordinates
        )

        # Outer head
        self.outer_head = nn.Sequential( 
            nn.Linear(head_input, 256),  
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),  # 3 coordinates
        )
 
    def _save(self, input_dict: dict, bracket_point_pred: torch.Tensor) -> None:
        """
        Writes the predicted points as JSON files.

        Each file will be named:
            {sample_name}_epoch{N}.json
        where N is how many times this function has been called for that sample.

        Args:
            input_dict (dict): Contains "name" (list of sample names without extension).
            bracket_point_pred (torch.Tensor): Predictions of shape [B, 9] (3 points * 3 coordinates).
        """
        # Initialize a counter the first time this is called
        if not hasattr(self, "_save_counter"):
            self._save_counter = defaultdict(int)

        os.makedirs(self.output_dir, exist_ok=True)

        names = input_dict["name"]
        preds = bracket_point_pred.detach().cpu().numpy()  # convert to numpy for JSON serialization

        for name, pred in zip(names, preds):
            # Increment epoch counter for this specific sample
            self._save_counter[name] += 1
            epoch_idx = self._save_counter[name]

            # Construct output filename
            filename = f"{name}_epoch{epoch_idx}.json"
            filepath = os.path.join(self.output_dir, filename)

            # Reshape predictions to 3 points
            bracket = pred[:3].tolist()
            incisal = pred[3:6].tolist()
            outer = pred[6:9].tolist()
            
            # Write coordinates to JSON
            with open(filepath, "w") as f:
                json.dump({
                    "bracket": bracket,
                    "incisal": incisal,
                    "outer": outer
                }, f, indent=4)


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
            if type(input_dict["name"]) == type(""): # then we are in test mode
                class_indices = torch.tensor([(int(input_dict["name"].split("_")[-1]) % 10) - 1]).long()
            else:
                class_indices = torch.tensor(  
                    [(int(x.split("_")[-1]) % 10) - 1 for x in input_dict["name"]],  
                    device=feat.device  
                ).long()
            class_indices = torch.clamp(class_indices, min=0, max=7)
            class_embeddings = self.embedder(class_indices)
            input_tensor = torch.cat([feat, class_embeddings], dim=1)
        else:
            input_tensor = feat 

        # Regress position of 3 points (bracket, incisal, outer), each with 3D coordinates
        bracket_pred = self.bracket_head(input_tensor)  # [B, 3]
        incisal_pred = self.incisal_head(input_tensor)  # [B, 3]
        outer_pred = self.outer_head(input_tensor)     # [B, 3]
        
        # Concatenate predictions for output (shape [B, 9])
        prediction = torch.cat([bracket_pred, incisal_pred, outer_pred], dim=1)  # [B, 9]
        
        if not self.training:
            out = {"bracket_point_pred": prediction}  # Add predictions to output (keep as [B, 9])
        else:
            out = {}
         
        # Compute MAE loss if ground truth available
        if "bracket" in input_dict:
            bracket_target = input_dict["bracket"].view_as(bracket_pred)
            incisal_target = input_dict["incisal"].view_as(incisal_pred)
            outer_target = input_dict["outer"].view_as(outer_pred)
            
            bracket_mae = nn.functional.l1_loss(bracket_pred, bracket_target)
            incisal_mae = nn.functional.l1_loss(incisal_pred, incisal_target)
            outer_mae = nn.functional.l1_loss(outer_pred, outer_target)
            
            bracket_cos_sim = nn.functional.cosine_similarity(bracket_pred, bracket_target).mean()
            incisal_cos_sim = nn.functional.cosine_similarity(incisal_pred, incisal_target).mean()
            outer_cos_sim = nn.functional.cosine_similarity(outer_pred, outer_target).mean()
            
            bracket_cos_dist = 1.0 - bracket_cos_sim
            incisal_cos_dist = 1.0 - incisal_cos_sim
            outer_cos_dist = 1.0 - outer_cos_sim
            
            mae = bracket_mae + incisal_mae + outer_mae
            cos_dist = (bracket_cos_dist + incisal_cos_dist + outer_cos_dist) / 3.0
            
            out["bracket_mae"] = bracket_mae
            out["incisal_mae"] = incisal_mae
            out["outer_mae"] = outer_mae
            out["bracket_cos_dist"] = bracket_cos_dist
            out["incisal_cos_dist"] = incisal_cos_dist
            out["outer_cos_dist"] = outer_cos_dist
            
            out["mae"] = mae
            out["cos_dist"] = cos_dist
            out["loss"] = (self.mae_weight * mae)
 
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
        output_dir: str = "output",
        use_refinement = False,
        freeze_backbone = False,
        freeze_coarse = False,  
    ): 
        super().__init__()  
            
        self.backbone = build_model(backbone)  
        self.alpha = alpha   
        self.output_dir = output_dir  
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