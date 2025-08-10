import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank,
             is_stage1b=False):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info(f"Start training stage {'1b' if is_stage1b else '1a'}")

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()
    xent = SupConLoss(device)
    
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    
    image_features = []
    labels = []
    views = [] # <-- Add list to store view labels
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            target_view_tensor = target_view.to(device)

            with torch.cuda.amp.autocast(enabled=True):
                image_feature = model(x=img, get_image=True)
                for i, v, img_feat in zip(target, target_view_tensor, image_feature):
                    labels.append(i)
                    views.append(v)
                    image_features.append(img_feat.cpu())

        labels_list = torch.stack(labels, dim=0).cuda()
        views_list = torch.stack(views, dim=0).cuda() # <-- Create tensor of views
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features, views

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        if scheduler is not None:
            scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            target_view_batch = views_list[b_list]
            image_features = image_features_list[b_list]

            with torch.cuda.amp.autocast(enabled=True):
                if is_stage1b:
                    text_features = model(label=target, get_text=True, view=target_view_batch)
                else:
                    text_features = model(label=target, get_text=True, view=None)

            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)

            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), b_list.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                lr = scheduler._get_lr(epoch)[0] if scheduler is not None else optimizer.param_groups[0]['lr']
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), i_ter + 1,
                                    loss_meter.avg, lr))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_stage{"1b" if is_stage1b else "1a"}_{epoch}.pth'))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_stage{"1b" if is_stage1b else "1a"}_{epoch}.pth'))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info(f"Stage {'1b' if is_stage1b else '1a'} running time: {total_time}")
