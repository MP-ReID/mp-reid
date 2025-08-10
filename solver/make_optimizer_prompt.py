import torch
import logging # 导入 logging 模块


# def make_optimizer_1stage(cfg, model):
#     logger = logging.getLogger("transreid.train")
#     params = []
#     keys = []
#     logger.info("--- Stage 1 Optimizer: Finding Trainable Parameters (expecting prompt_learner) ---")
#     trainable_param_count = 0
#     total_param_count = 0
#     for key, value in model.named_parameters():
#         total_param_count += value.numel()
#         if "prompt_learner" in key:
#             lr = cfg.SOLVER.STAGE1.BASE_LR
#             weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
#             params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#             keys += [key]
#             value.requires_grad_(True)
#             logger.info(f"  [Trainable] {key} (Size: {value.shape}, LR: {lr:.2e}, WD: {weight_decay:.2e})")
#             trainable_param_count += value.numel()
#         else:
#             value.requires_grad_(False)

#     logger.info(f"--- Stage 1 Optimizer: Found {len(keys)} trainable parameter groups.")
#     logger.info(f"--- Stage 1 Optimizer: Total trainable parameters: {trainable_param_count} / {total_param_count} ({trainable_param_count/total_param_count:.2%})")

#     if not params:
#         logger.warning("Warning: No parameters found for 'prompt_learner' in make_optimizer_1stage.")

#     if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'SGD':
#         optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE1.MOMENTUM)
#     elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'AdamW':
#         optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
#     else:
#         optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)
#     return optimizer
def make_optimizer_1stage(cfg, model, stage_name):
    stage_cfg = getattr(cfg.SOLVER, stage_name)
    
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        
        lr = stage_cfg.BASE_LR
        weight_decay = stage_cfg.WEIGHT_DECAY
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer_name = stage_cfg.OPTIMIZER_NAME
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=stage_cfg.MOMENTUM)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(params, amsgrad=stage_cfg.AMSGRAD)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(params)
        
    return optimizer

def make_optimizer_2astage(cfg, model, center_criterion):
    logger = logging.getLogger("transreid.train") # 获取 logger
    params = []
    keys = []
    logger.info("--- Stage 2a Optimizer: Finding Trainable Parameters ---")
    trainable_param_count = 0
    total_param_count = 0
    for key, value in model.named_parameters():
        total_param_count += value.numel()
        if value.requires_grad:
            value.requires_grad_(True)
            if "text_encoder" in key:
                value.requires_grad_(False)
                continue
            if "expert" in key:
                value.requires_grad_(False)
                continue

            lr = cfg.SOLVER.STAGE2.BASE_LR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
            if cfg.SOLVER.STAGE2.LARGE_FC_LR:
                if "classifier" in key or "arcface" in key:
                    lr = cfg.SOLVER.BASE_LR * 2
                    print('Using two times learning rate for fc ')
            
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
            logger.info(f"  [Trainable] {key} (Size: {value.shape}, LR: {lr:.2e}, WD: {weight_decay:.2e})")
            trainable_param_count += value.numel()
        # else: # 可选：记录冻结的参数
        #    logger.debug(f"  [Frozen] {key} (Size: {value.shape})") # 使用 debug 级别

    logger.info(f"--- Stage 2a Optimizer: Found {len(keys)} trainable parameter groups.")
    logger.info(f"--- Stage 2a Optimizer: Total trainable parameters: {trainable_param_count} / {total_param_count} ({trainable_param_count/total_param_count:.2%})")

    if not params:
         logger.warning("Warning: No trainable parameters found for the optimizer in stage 2a.") # 使用 warning

    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center

def make_optimizer_2bstage(cfg, model, center_criterion):
    logger = logging.getLogger("transreid.train") # 获取 logger
    params = []
    keys = []
    logger.info("--- Stage 2b Optimizer: Finding Trainable Parameters ---")
    trainable_param_count = 0
    total_param_count = 0
    for key, value in model.named_parameters():
        total_param_count += value.numel()
        if "gate" in key:
            value.requires_grad_(True)
        elif "image_encoder" in key and "experts" not in key:
            value.requires_grad_(True)
        else:
            value.requires_grad_(False)
            continue
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
            
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
        logger.info(f"  [Trainable] {key} (Size: {value.shape}, LR: {lr:.2e}, WD: {weight_decay:.2e})")
        trainable_param_count += value.numel()

    logger.info(f"--- Stage 2b Optimizer: Found {len(keys)} trainable parameter groups.")
    logger.info(f"--- Stage 2b Optimizer: Total trainable parameters: {trainable_param_count} / {total_param_count} ({trainable_param_count/total_param_count:.2%})")

    if not params:
         logger.warning("Warning: No trainable parameters found for the optimizer in stage 2b.") # 使用 warning

    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center

def make_optimizer_lora(cfg, model, center_criterion):
    logger = logging.getLogger("transreid.train") # 获取 logger
    params = []
    keys = []
    logger.info("--- LoRA Stage Optimizer: Finding Trainable Parameters ---")
    trainable_param_count = 0
    total_param_count = 0

    for key, value in model.named_parameters():
        total_param_count += value.numel()
        # Collect all parameters that PEFT left as trainable (LoRA params + potentially others)
        if value.requires_grad:
            # Read parameters from SOLVER.LORA block
            lr = cfg.SOLVER.LORA.BASE_LR
            weight_decay = cfg.SOLVER.LORA.WEIGHT_DECAY
            if "bias" in key:
                # Apply specific LR/WD for bias if needed, check if LoRA applies bias
                # Assuming LORA block also has BIAS factors if needed, else use default LORA WD
                bias_lr_factor = getattr(cfg.SOLVER.LORA, 'BIAS_LR_FACTOR', 1.0) # Default to 1 if not defined
                lr = cfg.SOLVER.LORA.BASE_LR * bias_lr_factor
                weight_decay = getattr(cfg.SOLVER.LORA, 'WEIGHT_DECAY_BIAS', cfg.SOLVER.LORA.WEIGHT_DECAY)

            # Check for large_fc_lr specifically in LORA config if needed
            large_fc_lr_lora = getattr(cfg.SOLVER.LORA, 'LARGE_FC_LR', False)
            if large_fc_lr_lora and ("classifier" in key or "arcface" in key):
                 # Check if classifier/arcface are intended to be trained during LoRA stage
                 lr = cfg.SOLVER.LORA.BASE_LR * 2 # Assuming factor is 2x
                 # print(f'Using 2x LR for {key} in LoRA stage') # Optional print

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
            # Log trainable parameters for LoRA stage
            logger.info(f"  [Trainable] {key} (Size: {value.shape}, LR: {lr:.2e}, WD: {weight_decay:.2e})")
            trainable_param_count += value.numel()

    logger.info(f"--- LoRA Stage Optimizer: Found {len(keys)} trainable parameter groups.")
    logger.info(f"--- LoRA Stage Optimizer: Total trainable parameters: {trainable_param_count} / {total_param_count} ({trainable_param_count/total_param_count:.2%})")

    if not params:
        logger.warning("Warning: No trainable parameters found for the optimizer in LoRA stage.")

    # Use optimizer settings from SOLVER.LORA block
    optimizer_name = cfg.SOLVER.LORA.OPTIMIZER_NAME
    if optimizer_name == 'SGD':
        momentum = getattr(cfg.SOLVER.LORA, 'MOMENTUM', 0.9) # Provide default if not set
        optimizer = getattr(torch.optim, optimizer_name)(params, momentum=momentum)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.LORA.BASE_LR, weight_decay=cfg.SOLVER.LORA.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(params)

    # Center loss optimizer LR might also need its own config? Using STAGE2 for now.
    # Consider adding SOLVER.LORA.CENTER_LR if needed.
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center