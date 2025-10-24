"""
Configuration file for Bracket Point Prediction
"""

_base_ = ["../_base_/default_runtime.py"]

# -----------------------------
# Misc settings
# -----------------------------
batch_size = 4        # total batch size across all GPUs
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
# Model settings
# -----------------------------
model = dict(
    type="SimpleBracketPredictor",
    in_channels=3,  # model will use only coordinates as input
    hidden_dim=128,
    criteria=[
        dict(type="MSELoss"),  # Mean Squared Error for regression
    ],
)

# -----------------------------
# Optimizer & Scheduler
# -----------------------------
epoch = 100
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="MultiStepLR",
    milestones=[10, 15],
    gamma=0.1,
)

# -----------------------------
# Dataset settings
# -----------------------------
dataset_type = "BracketPointDataset"
data_root = "/work/grana_maxillo/Mlugli/brackets_melted/flattened"

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=["x", "y", "z"],

    train=dict(
        type=dataset_type,
        split="train",
        point_count = 2048,
        data_root=data_root,
        transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "bracket_point"],  # must be a list
                feat_keys=["coord"],              # features for the model
            ),
        ],
        test_mode=False
    ),

    val=dict(
        type=dataset_type,
        split="val",
        point_count = 2048,
        data_root=data_root,
        transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord", "bracket_point"],
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        split="test",
        point_count = 2048,
        data_root=data_root,
        transform=[
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=["coord"],
                feat_keys=["coord"],
            ),
        ],
        test_mode=True,
    ),
)

# -----------------------------
# Hooks (default)
# -----------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="DisplacementEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
