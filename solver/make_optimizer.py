import torch

def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center

def make_optimizer_1stage(cfg, model, stage_name):
    stage_cfg = getattr(cfg.SOLVER, stage_name)
    
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        
        lr = stage_cfg.BASE_LR
        weight_decay = stage_cfg.WEIGHT_DECAY
        
        if "bias" in key:
            lr = stage_cfg.BASE_LR * stage_cfg.BIAS_LR_FACTOR
            weight_decay = stage_cfg.WEIGHT_DECAY_BIAS
            
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer_name = stage_cfg.OPTIMIZER_NAME
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=stage_cfg.MOMENTUM)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(params, amsgrad=stage_cfg.AMSGRAD)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(params)
        
    return optimizer