from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# MoE Parameter
_C.MODEL.MOE = CN()
_C.MODEL.MOE.ENABLED = False
_C.MODEL.MOE.NUM_EXPERTS = 0
_C.MODEL.MOE.TOP_K = 0
_C.MODEL.MOE.MOE_LAYERS = 0
_C.MODEL.MOE.DROPOUT = 0.0
_C.MODEL.MOE.FREEZE_EXCEPT_GATE = False
_C.MODEL.MOE.MODEL_PATH_LIST = []
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
_C.DATASETS.EXP_SETTING = ('cctv_ir_cctv_rgb')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234
_C.SOLVER.MARGIN = 0.3

# stage1
# ---------------------------------------------------------------------------- #
# Name of optimizer
_C.SOLVER.STAGE1 = CN()

_C.SOLVER.STAGE1.IMS_PER_BATCH = 64

_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE1.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE1.BASE_LR = 3e-4
# Momentum
_C.SOLVER.STAGE1.MOMENTUM = 0.9

# Settings of weight decay
_C.SOLVER.STAGE1.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE1.WEIGHT_DECAY_BIAS = 0.0005

# warm up factor
_C.SOLVER.STAGE1.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE1.LR_MIN = 0.000016

_C.SOLVER.STAGE1.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE1.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE1.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE1.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE1.LOG_PERIOD = 100
# epoch number of validation
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
# _C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.EVAL_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Solver
# stage1a
# ---------------------------------------------------------------------------- #
_C.SOLVER.STAGE1A = CN()
_C.SOLVER.STAGE1A.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1A.OPTIMIZER_NAME = "Adam"
_C.SOLVER.STAGE1A.MAX_EPOCHS = 100
_C.SOLVER.STAGE1A.BASE_LR = 3e-4
_C.SOLVER.STAGE1A.MOMENTUM = 0.9
_C.SOLVER.STAGE1A.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE1A.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.STAGE1A.WARMUP_FACTOR = 0.01
_C.SOLVER.STAGE1A.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE1A.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE1A.LR_MIN = 0.000016
_C.SOLVER.STAGE1A.WARMUP_ITERS = 500
_C.SOLVER.STAGE1A.WARMUP_METHOD = "linear"
_C.SOLVER.STAGE1A.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE1A.COSINE_SCALE = 30
_C.SOLVER.STAGE1A.CHECKPOINT_PERIOD = 10
_C.SOLVER.STAGE1A.LOG_PERIOD = 100
_C.SOLVER.STAGE1A.EVAL_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Solver
# stage1b
# ---------------------------------------------------------------------------- #
_C.SOLVER.STAGE1B = CN()
_C.SOLVER.STAGE1B.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1B.OPTIMIZER_NAME = "Adam"
_C.SOLVER.STAGE1B.MAX_EPOCHS = 100
_C.SOLVER.STAGE1B.BASE_LR = 3e-4
_C.SOLVER.STAGE1B.MOMENTUM = 0.9
_C.SOLVER.STAGE1B.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE1B.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.STAGE1B.WARMUP_FACTOR = 0.01
_C.SOLVER.STAGE1B.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE1B.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE1B.LR_MIN = 0.000016
_C.SOLVER.STAGE1B.WARMUP_ITERS = 500
_C.SOLVER.STAGE1B.WARMUP_METHOD = "linear"
_C.SOLVER.STAGE1B.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE1B.COSINE_SCALE = 30
_C.SOLVER.STAGE1B.CHECKPOINT_PERIOD = 10
_C.SOLVER.STAGE1B.LOG_PERIOD = 100
_C.SOLVER.STAGE1B.EVAL_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Solver
# stage2
# ---------------------------------------------------------------------------- #
_C.SOLVER.STAGE2 = CN()

_C.SOLVER.STAGE2.IMS_PER_BATCH = 64
# Name of optimizer
_C.SOLVER.STAGE2.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE2.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE2.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.STAGE2.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.STAGE2.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.STAGE2.MOMENTUM = 0.9
# Margin of triplet loss
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.STAGE2.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.STAGE2.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.STAGE2.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE2.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.STAGE2.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STAGE2.STEPS = (40, 70)
# warm up factor
_C.SOLVER.STAGE2.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE2.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE2.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE2.LR_MIN = 0.000016


_C.SOLVER.STAGE2.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE2.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE2.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE2.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE2.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE2.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.STAGE2.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch

# ---------------------------------------------------------------------------- #
# Solver
# lora
# ---------------------------------------------------------------------------- #
_C.SOLVER.LORA = CN()

_C.SOLVER.LORA.LORA_R = 8
_C.SOLVER.LORA.LORA_ALPHA = 16
_C.SOLVER.LORA.LORA_DROPOUT = 0.1

_C.SOLVER.LORA.IMS_PER_BATCH = 64

_C.SOLVER.LORA.OPTIMIZER_NAME = "Adam"

# Base learning rate
_C.SOLVER.LORA.BASE_LR = 0.00001
_C.SOLVER.LORA.WARMUP_LR_INIT = 0.000001
_C.SOLVER.LORA.LR_MIN = 0.000001

_C.SOLVER.LORA.WARMUP_METHOD = 'linear'
# Settings of weight decay
_C.SOLVER.LORA.WEIGHT_DECAY = 0.0001
_C.SOLVER.LORA.WEIGHT_DECAY_BIAS = 0.0001
# Number of max epoches
_C.SOLVER.LORA.MAX_EPOCHS = 30
#  warm up epochs
_C.SOLVER.LORA.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.LORA.WARMUP_METHOD = "linear"
# epoch number of saving checkpoints
_C.SOLVER.LORA.CHECKPOINT_PERIOD = 30
# iteration of display training log
_C.SOLVER.LORA.LOG_PERIOD = 50
# epoch number of validation
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
# _C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.LORA.EVAL_PERIOD = 5

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False

# --- Add TTA and TTPT settings ---
# Whether to enable Test-Time Augmentation (e.g., horizontal flip) for query images
_C.TEST.TTA_ENABLED = False

# Test-Time Prompt Tuning Settings
_C.TEST.TTPT = CN()
# Whether to enable Test-Time Prompt Tuning (Option B / CLIP-style evaluation)
_C.TEST.TTPT.ENABLED = False
# Learning rate for the TTPT optimizer (optimizing PromptLearner.cls_ctx)
_C.TEST.TTPT.LR = 0.001
# Number of optimization steps for TTPT per query image
_C.TEST.TTPT.STEPS = 5
# Temperature for the softmax in the entropy loss during TTPT
_C.TEST.TTPT.TEMPERATURE = 0.07
# --- End TTA and TTPT settings ---

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
