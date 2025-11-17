

"""  
Configuration file for Bracket Point Prediction with Voxel-based Backbone
"""  
  
_base_ = ["../_base_/default_runtime.py"]
# -----------------------------
# Misc settings
# -----------------------------
batch_size = 32
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
    backbone=dict(                          #  Backbone: Point Transformer V3 (v1m1)
        type="PT-v3m1",                     # as in cls-ptv3-v1m1-0-base
        in_channels=6,                      # 3 xyz 
        enc_channels=(16, 32, 48, 64, 128),
        enc_num_head=(1, 2, 3, 4, 8),
        dec_channels=(32, 32, 64, 96),
        dec_num_head=(2, 2, 4, 6),
        enable_flash=False,                 # True if AMPERE gpu arch
        cls_mode=True,
    ),
    backbone_out_channels=128,
    save_predictions = False,
    # output_dir will be unused if save_predictions is not set.
    output_dir = "/work/grana_maxillo/Mlugli/brackets_melted/model_predictions/json",
    output_dim=3,
    alpha=0.4, # Trying adding cosine similarity as an extra metric to improve precision
    class_embedding=False, # give class (FDI_index % 10) embedding to head.
)

# -----------------------------
# Optimizer & Scheduler
# -----------------------------
epoch = 40
eval_epoch = 40 # Set equal to epoch for single training run
clip_grad = 1.0

optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.005)  
scheduler = dict(
    type="OneCycleLR",  
    max_lr=optimizer["lr"],  # References the optimizer's lr  
    pct_start=0.20,
    anneal_strategy="cos", 
    div_factor=10.0,
    final_div_factor=100.0,
)

# optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
# scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)

# -----------------------------
# Dataset settings
# -----------------------------  
dataset_type = "BracketPointDataset"  
data_root = "/work/grana_maxillo/Mlugli/Brackets"  
fold = 6 # Fold to use
debased=False # Use debased data
feat_keys = ["coord", "normal"]
grid_size = 0.005

data = dict(
    train=dict(
        type=dataset_type,  
        split="train",
        debug=False,
        data_root=data_root,
        fold=fold,
        debased=debased,
        plot=False,
        transform=[
            dict(
                type='CustomRandomRotate',
                angle=[-0.05, 0.05],
                center=[0, 0, 0],
                axis='z',
                p=0.5),
            dict(type='CustomRandomRotate', angle=[-0.1, 0.1], axis='x', p=0.5),
            dict(type='CustomRandomRotate', angle=[-0.1, 0.1], axis='y', p=0.5),
            dict(type='CustomRandomScale', scale=[0.9, 1.1]),
            dict(type='CustomRandomFlip', p=0.5),
            dict(
                type='CustomRandomShift',
                shift=((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05))),
            #dict(type='RandomJitter', sigma=0.00125, clip=0.005),
            dict(
                type='GridSample',
                grid_size=grid_size,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=['coord', 'grid_coord', 'bracket', 'name'],
                feat_keys=feat_keys)
        ],  
        test_mode=False
    ),  
  
    val=dict(  
        type=dataset_type,
        split="val",
        data_root=data_root,
        fold=fold,
        debased=debased,
        transform=[
            #dict(type="Update", keys_dict={"index_valid_keys": ["coord"]}),  # Add this line  
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,  # This generates grid_coord
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",  
                keys=["coord", "grid_coord", "bracket", "name"],
                feat_keys=feat_keys,  
            ),
        ],
        test_mode=False,  
    ),
    # inference
    test=dict(  
        type=dataset_type,
        data_root=data_root,
        fold=fold,
        debased=debased,
        split="test",
        transform=[
            #dict(type="Update", keys_dict={"index_valid_keys": ["coord"]}),  # Add this line
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size, 
                hash_type="fnv", 
                mode="test",
                return_grid_coord=True,  
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=feat_keys,
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
