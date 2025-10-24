"""  
Configuration file for Bracket Point Prediction with Voxel-based Backbone  
"""  
  
_base_ = ["../_base_/default_runtime.py"]  
  
# -----------------------------
# Misc settings
# -----------------------------
batch_size = 2
num_worker = 4
mix_prob = 0
empty_cache = False  
enable_amp = True  

# -----------------------------
# Wandb settings
# -----------------------------
enable_wandb = True    
wandb_project = "bracket_point_prediction"  
wandb_key = "aa5065e448806f5c4fc22ae91586d8ecd5183c9c"   
  
# -----------------------------
# Model settings, Using SpUNet backbone
# Make sure to use cls_mode=True so that the 
# decoding part of the backbone architecture (producing
# the segmentation) is not used.
# -----------------------------

model = dict(
    type="VoxelBracketPredictor",  
    backbone=dict(  
        type="SpUNet-v1m1",  
        in_channels=3,  # xyz coordinates only  
        num_classes=0,  
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        cls_mode = True,
    ),  
    backbone_out_channels=256,
    save_predictions = True,
    output_dir = "/work/grana_maxillo/Mlugli/brackets_melted/model_predictions/json",
    output_dim=3,
)  

# -----------------------------
# Optimizer & Scheduler
# Trying Lorenzo's scheduler and optimizer configs.
# -----------------------------
epoch = 100
eval_epoch = 100  # Set equal to epoch for single training run

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)
  
# -----------------------------
# Dataset settings
# -----------------------------  
dataset_type = "BracketPointDataset"  
data_root = "/work/grana_maxillo/Mlugli/brackets_melted/flattened"  

data = dict(  
    train=dict(  
        type=dataset_type,  
        split="train",  
        data_root=data_root,
        plot=False,
        transform=[  
            #dict(type="CenterShift", apply_z=True),  
            #dict(type="RandomRotate", angle=[-1, 1], axis="z", p=0.5),  
            #dict(type="RandomScale", scale=[0.9, 1.1]),  
            dict(  
                type="GridSample",
                grid_size=0.005,  
                hash_type="fnv",  
                mode="train", 
                return_grid_coord=True,  # This generates grid_coord  
            ),  
            #dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),  
            dict(  
                type="Collect",  
                keys=["coord", "grid_coord", "bracket_point", "name"],  # Include grid_coord  
                feat_keys=["coord"],  
            ),  
        ],  
        test_mode=False  
    ),  
  
    val=dict(  
        type=dataset_type,  
        split="val",  
        data_root=data_root,
        transform=[  
            #dict(type="CenterShift", apply_z=True),  
            dict(  
                type="GridSample",  
                grid_size=0.005,  
                hash_type="fnv",  
                mode="train",  
                return_grid_coord=True,  # This generates grid_coord  
            ),  
            #dict(type="CenterShift", apply_z=False),  
            dict(type="ToTensor"),  
            dict(  
                type="Collect",  
                keys=["coord", "grid_coord", "bracket_point", "name"],
                feat_keys=["coord"],  
            ),  
        ],  
        test_mode=False,  
    ),  
)
  
# -----------------------------
# Hooks
# Default ones + DisplacementEvaluator that
# computes the loss of the regression task.
# -----------------------------
hooks = [  
    dict(type="CheckpointLoader"),  
    dict(type="IterationTimer", warmup_iter=2),  
    dict(type="InformationWriter"),  
    dict(type="DisplacementEvaluator"), 
    dict(type="CheckpointSaver", save_freq=None),  
]
