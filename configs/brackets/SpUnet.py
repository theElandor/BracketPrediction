"""  
Configuration file for Bracket Point Prediction with Voxel-based Backbone  
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
    save_predictions = False,
    output_dir = "/work/grana_maxillo/Mlugli/brackets_melted/model_predictions/json",
    output_dim=3,
    mode = "offset"
)

# -----------------------------
# Optimizer & Scheduler
# -----------------------------
epoch = 50
eval_epoch = 50  # Set equal to epoch for single training run
clip_grad = 1.0



optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.005)  
scheduler = dict(
    type="CosineAnnealingLR",  
    total_steps=epoch,  # Will be set automatically by the trainer  
    eta_min=0,  
)

# optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
# scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)
  
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
            dict(
                type='CustomRandomRotate',
                angle=[-0.1, 0.1],
                center=[0, 0, 0],
                axis='z',
                p=0.5),
            dict(type='CustomRandomScale', scale=[0.95, 1.05]),
            dict(
                type='CustomRandomShift',
                shift=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))),
            dict( # dropout just drops some points from "coord",
                # no need for custom modifications
                type='RandomDropout',
                dropout_ratio=0.5,
                dropout_application_ratio=0.5),
            dict(
                type='GridSample',
                grid_size=0.005,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=['coord', 'grid_coord', 'bracket_point', 'facial_point', 'name'],
                feat_keys=['coord'])
        ],  
        test_mode=False
    ),  
  
    val=dict(  
        type=dataset_type,  
        split="val",  
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict={"index_valid_keys": ["coord"]}),  # Add this line  
            dict(  
                type="GridSample",  
                grid_size=0.005,  
                hash_type="fnv",  
                mode="train",  
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),  
            dict(
                type="Collect",  
                keys=["coord", "grid_coord", "bracket_point", "facial_point", "name"],
                feat_keys=["coord"],  
            ),  
        ],  
        test_mode=False,  
    ),
    test=dict(  
        type=dataset_type,  
        data_root=data_root,  
        split="test",
        transform=[
            # Base transforms before test processing  
            dict(type="Update", keys_dict={"index_valid_keys": ["coord"]}),  # Add this line
        ],
        test_mode=True,  
        test_cfg=dict(  
            voxelize=dict(  
                type="GridSample",  
                grid_size=0.005,  
                hash_type="fnv",  
                mode="test",  
                return_grid_coord=True,  
            ),  
            crop=None,  
            post_transform=[  
                dict(type="ToTensor"),  
                dict(  
                    type="Collect",  
                    keys=("coord", "grid_coord", "facial_point", "index"),  
                    feat_keys=("coord",),  
                ),  
            ],  
            aug_transform=[  
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]  
            ],  
        ),  
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
test = dict(type="BracketTester")