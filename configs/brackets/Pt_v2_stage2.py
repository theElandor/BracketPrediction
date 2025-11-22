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
enable_wandb = False
wandb_project = "bracket_point_prediction"

# -----------------------------
# Make sure to use cls_mode=True so that the 
# decoding part of the backbone architecture (producing
# the segmentation) is not used.
# -----------------------------

model = dict(
    type="VoxelBracketPredictor_v2",
    backbone=dict(                          
        type="PT-v3m1", 
        in_channels=3,                      # 3 xyz
        enc_channels=(16, 32, 48, 64, 128),
        enc_num_head=(1, 2, 3, 4, 8),
        dec_channels=(32, 32, 64, 96),
        dec_num_head=(2, 2, 4, 6),
        enable_flash=False,          # True if AMPERE gpu arch
        cls_mode=True,
    ),
    backbone_out_channels=128,
    save_predictions = False,
    output_dim=3,
    alpha=0.4, # Trying adding cosine similarity as an extra metric to improve precision
    freeze_backbone=True,
    freeze_coarse=True,
    use_refinement=True
)

# -----------------------------
# Optimizer & Scheduler
# -----------------------------
epoch = 30
eval_epoch = 30 # Set equal to epoch for single training run
clip_grad = 1.0

optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.005)  
scheduler = dict(  
    type="OneCycleLR",  
    max_lr=optimizer["lr"],  # References the optimizer's lr  
    pct_start=0.10,
    anneal_strategy="cos", 
    div_factor=10.0,
    final_div_factor=100.0,
)

# -----------------------------
# Dataset settings
# -----------------------------  
dataset_type = "BracketPointDataset"  
data_root = "/work/grana_maxillo/Mlugli/Brackets"  
fold = 6 # Fold to use
debased=False # Use debased data
feat_keys = ["coord"]
grid_size = 0.005
# Custom augmentations are the same as the standard
# pointcept augmentations but they apply the transform
# also to the GT point, so that it is rotated/flipped/shifted
# together with the coordinates.
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
                    keys=("coord", "grid_coord", "index", "name"),
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
    dict(type="CheckpointLoader",
        keywords = "module.",
        replacement = "module.",
    ),  
    dict(type="IterationTimer", warmup_iter=2),  
    dict(type="InformationWriter"),  
    dict(type="DisplacementEvaluator"), 
    dict(type="CheckpointSaver", save_freq=None),  
]
test = dict(type="BracketTester")