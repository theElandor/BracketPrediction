"""    
Same as sp_map.py but takes 6 channels in input.
"""

_base_ = ["../_base_/default_runtime.py"]

# -----------------------------  
# Misc settings
# -----------------------------  
batch_size = 16
num_worker = 4
mix_prob = 0
empty_cache = False
enable_amp = True
  
# -----------------------------
# Wandb settings
# -----------------------------
enable_wandb = True
wandb_project = "bracket_point_prediction"

# -----------------------------
# Model settings
# ----------------------------- 
model = dict(      
    type="HeatmapRegressor",      
    backbone=dict(      
        type="SpUNet-v1m1",      
        in_channels=6,  # xyz only    
        num_classes=0,  # Set to 0 for regression tasks  
        channels=(32, 64, 128, 256, 256, 128, 64, 64),  
        layers=(2, 3, 4, 6, 2, 2, 2, 2),  
    ),
    backbone_out_channels=64,  # Updated to match SpUNet's last channel  
)
  
# -----------------------------  
# Optimizer & Scheduler
# -----------------------------  
epoch = 80
eval_epoch = 80
clip_grad = 1.0

optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.10,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# -----------------------------  
# Dataset settings
# -----------------------------    
dataset_type = "BracketMapDataset"
data_root = "/work/grana_maxillo/Mlugli/BracketsV1"
feat_keys = ["coord", "normal"]
grid_size = 0.005

data = dict(
    train=dict(
        type=dataset_type,
        split="train", 
        debug=False,
        data_root=data_root,
        transform=[
            dict(type='RandomRotate', angle=[-0.1, 0.1], axis='z', p=0.5),
            dict(type='RandomRotate', angle=[-0.1, 0.1], axis='x', p=0.5),
            dict(type='RandomRotate', angle=[-0.1, 0.1], axis='y', p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomShift', shift=((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))),
            dict(
                type='GridSample',
                grid_size=grid_size,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=['coord', 'grid_coord', 'name', 'segment'],
                feat_keys=feat_keys)
        ],    
        test_mode=False
    ),
    
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[  
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),  
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "grid_coord", "segment", "origin_segment", "inverse", "name"],
                feat_keys=feat_keys,
            ),
        ],
        test_mode=False,
    ),
 
    test=dict(    
        type="BracketMapDataset",
        split="test",
        data_root=data_root,
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_inverse = True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index", "inverse"), feat_keys=feat_keys),
            ],
            aug_transform=[
                [dict(type='RandomRotate', angle=[-0.1, 0.1], axis='z', p=0.5)],
                [dict(type='RandomRotate', angle=[-0.1, 0.1], axis='x', p=0.5)],
                [dict(type='RandomRotate', angle=[-0.1, 0.1], axis='y', p=0.5)],
            ],
        ),
    ),
)  
   
# -----------------------------  
# Hooks  
# -----------------------------  
hooks = [  
    dict(type="CheckpointLoader"),    
    dict(type="IterationTimer", warmup_iter=2),    
    dict(type="InformationWriter"),  
    dict(type="HeatmapEvaluator"),  
    dict(type="CheckpointSaver", save_freq=None),    
]  
  
test = dict(  
    type="HeatmapTester",  
    verbose=True  
)
