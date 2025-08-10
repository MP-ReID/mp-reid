import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from model.clip.model import load_balancing_loss_func
import numpy as np

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             max_epochs,
             log_period,
             checkpoint_period,
             eval_period):
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = max_epochs

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = torch.amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with torch.amp.autocast('cuda', enabled=True):
                text_feature = model(label = l_list, get_text = True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    # Set coefficient for auxiliary loss
    load_balance_loss_coeff = 0.01 # You might want to move this to config later
    logger.info(f"Using Load Balancing Loss Coefficient: {load_balance_loss_coeff}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step(epoch)

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with torch.amp.autocast('cuda', enabled=True):
                model_outputs = model(x = img, label = target,
                                      cam_label=target_cam,
                                      view_label=target_view)

                if len(model_outputs) == 4:
                    scores, feats_all, image_features_proj, router_logits = model_outputs
                elif len(model_outputs) == 3: # Fallback for non-MoE or if logits aren't returned
                     scores, feats_all, image_features_proj = model_outputs
                     router_logits = None
                else:
                    raise ValueError(f"Unexpected number of outputs from model: {len(model_outputs)}")

                score = scores[0]
                feat = feats_all[1]

                logits_i2t = image_features_proj @ text_features.t()

                loss = loss_fn(score, feat, target, target_cam, logits_i2t)

                if router_logits is not None and cfg.MODEL.MOE.ENABLED and load_balance_loss_coeff > 0:
                    l_aux = 0
                    num_moe_layers_with_logits = router_logits.shape[0]
                    if num_moe_layers_with_logits > 0:
                        for layer_logits in router_logits:
                            l_aux += load_balancing_loss_func(layer_logits, cfg.MODEL.MOE.TOP_K)
                        l_aux = l_aux / num_moe_layers_with_logits
                        loss = loss + load_balance_loss_coeff * l_aux
                    else:
                        logger.warning("router_logits received but has 0 layers?")

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits_i2t.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log_msg = "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}" \
                          .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                  loss_meter.avg, acc_meter.avg, current_lr)
                if 'l_aux' in locals() and l_aux is not None and load_balance_loss_coeff > 0:
                     log_msg += f", AuxLoss: {load_balance_loss_coeff * l_aux.item():.3f}"
                logger.info(log_msg)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR,cfg.DATASETS.EXP_SETTING, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR,cfg.DATASETS.EXP_SETTING, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, pid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(x=img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, pid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(x=img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

# 新增函数，用于实现 TTA + TTPT + CLIP-style Evaluation (Option B)
def do_inference_ttpt_clipstyle(cfg,
                                model,
                                val_loader,
                                num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test_ttpt_clipstyle")
    logger.info("Enter inferencing with TTA, TTPT (CLIP-style Evaluation - Option B)")

    # --- 配置参数 ---
    tta_enabled = cfg.TEST.get('TTA_ENABLED', True)
    ttpt_enabled = cfg.TEST.TTPT.get('ENABLED', True)
    ttpt_lr = cfg.TEST.TTPT.get('LR', 0.001)
    ttpt_steps = cfg.TEST.TTPT.get('STEPS', 5)
    ttpt_temperature = cfg.TEST.TTPT.get('TEMPERATURE', 0.07)
    feat_norm = cfg.TEST.FEAT_NORM

    if tta_enabled:
        logger.info("Test Time Augmentation (TTA) enabled.")
    if ttpt_enabled:
        logger.info(f"Test Time Prompt Tuning (TTPT) enabled: LR={ttpt_lr}, Steps={ttpt_steps}, Temp={ttpt_temperature}")

    # --- 模型和数据准备 ---
    if device:
        if torch.cuda.device_count() > 1:
            logger.info(f'Using {torch.cuda.device_count()} GPUs for inference')
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        model.to(device)
    else:
        model_without_ddp = model

    model.eval()

    # --- 存储特征、PID、CamID ---
    query_features = []
    query_pids = []
    query_camids = []
    gallery_features = []
    gallery_pids = []
    gallery_camids = []

    # --- 获取 TTPT 所需模块 ---
    if ttpt_enabled:
        try:
            prompt_learner = model_without_ddp.prompt_learner
            text_encoder = model_without_ddp.text_encoder
            tokenized_prompts = prompt_learner.tokenized_prompts.to(device)
            num_classes = prompt_learner.num_class
            all_class_labels = torch.arange(num_classes, device=device)
            logger.info("Successfully accessed PromptLearner and TextEncoder for TTPT.")
        except AttributeError as e:
            logger.error(f"Failed to get modules needed for TTPT: {e}. Disabling TTPT.")
            ttpt_enabled = False


    # --- 迭代数据加载器 ---
    logger.info("Starting feature extraction...")
    start_time = time.time()
    processed_samples = 0

    for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
        img = img.to(device)
        pid = np.asarray(pid)
        camid = np.asarray(camid)

        is_query = (processed_samples < num_query)

        with torch.no_grad():
            # --- 处理 Query ---
            if is_query:
                # 1. TTA (Test-Time Augmentation)
                img_feat_list = []
                feat_dim = model_without_ddp.in_planes_proj
                full_feat_orig = model(x=img, cam_label=camids.to(device) if cfg.MODEL.SIE_CAMERA else None, view_label=target_view.to(device) if cfg.MODEL.SIE_VIEW else None)
                img_feat_proj_orig = full_feat_orig[:, -feat_dim:]
                img_feat_list.append(img_feat_proj_orig)

                if tta_enabled:
                    img_flipped = torch.flip(img, [3])
                    full_feat_flipped = model(x=img_flipped, cam_label=camids.to(device) if cfg.MODEL.SIE_CAMERA else None, view_label=target_view.to(device) if cfg.MODEL.SIE_VIEW else None)
                    img_feat_proj_flipped = full_feat_flipped[:, -feat_dim:]
                    img_feat_list.append(img_feat_proj_flipped)

                # 聚合 TTA 特征 (平均)
                img_feat_agg = torch.stack(img_feat_list, dim=0).mean(dim=0)
                if feat_norm:
                    img_feat_agg = F.normalize(img_feat_agg, p=2, dim=1)


                # 2. TTPT (Test-Time Prompt Tuning)
                if ttpt_enabled:
                    # 保存原始 prompt 状态
                    initial_prompt_state = prompt_learner.cls_ctx.data.clone()

                    # 启用 prompt 上下文向量的梯度
                    prompt_learner.cls_ctx.requires_grad_(True)
                    # 冻结模型其他部分 (eval模式已大部分冻结，这里确保)
                    for name, param in model_without_ddp.named_parameters():
                        if 'prompt_learner.cls_ctx' not in name:
                            param.requires_grad_(False)

                    # 设置 TTPT 优化器
                    optimizer_ttp = torch.optim.AdamW([prompt_learner.cls_ctx], lr=ttpt_lr)

                    # --- TTPT 优化循环 --- 
                    # 将前向计算、loss计算、反向传播和优化器步骤放入 enable_grad 块
                    with torch.enable_grad():
                        for step in range(ttpt_steps):
                            # 生成当前所有类别的文本特征
                            prompts = prompt_learner(all_class_labels)
                            text_features_all = text_encoder(prompts, tokenized_prompts)

                            # 计算图文相似度 (使用聚合后的图像特征)
                            # img_feat_agg 不需要梯度，text_features_all 需要
                            similarity = img_feat_agg @ text_features_all.t()

                            # 计算熵损失
                            probs = F.softmax(similarity / ttpt_temperature, dim=-1)
                            loss_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                            loss = loss_entropy

                            optimizer_ttp.zero_grad()
                            loss.backward()
                            optimizer_ttp.step()

                    # 冻结 prompt 梯度（在 enable_grad 块外部完成）
                    prompt_learner.cls_ctx.requires_grad_(False)
                    # --- TTPT 优化结束 ---
                    
                    # --- 生成最终文本特征 (在 no_grad 或 enable_grad 外部) ---
                    # 这一步不需要梯度，可以在默认上下文(no_grad)中进行
                    final_prompts = prompt_learner(all_class_labels)
                    final_text_features = text_encoder(final_prompts, tokenized_prompts)
                    if feat_norm:
                        final_text_features = F.normalize(final_text_features, p=2, dim=1)

                    # 获取最相似的文本特征 (需要优化循环中计算的 similarity)
                    # 需要在 enable_grad 块结束后再计算这个
                    with torch.no_grad(): # 确保 argmax 不会尝试记录梯度
                       most_similar_idx = similarity.argmax(dim=1)
                    query_feat = final_text_features[most_similar_idx]

                    # 恢复原始 prompt 状态
                    prompt_learner.cls_ctx.data.copy_(initial_prompt_state)

                else:
                    logger.warning("TTPT is disabled, cannot perform CLIP-style evaluation (Option B). Skipping query sample.")
                    processed_samples += img.shape[0]
                    continue

                # 存储查询结果
                query_features.append(query_feat.cpu())
                query_pids.extend(pid)
                query_camids.extend(camid)

            # --- 处理 Gallery ---
            else:
                # 直接提取图像特征 (整个特征，包括非投影部分，与原始 do_inference 保持一致)
                gallery_feat = model(x=img, cam_label=camids.to(device) if cfg.MODEL.SIE_CAMERA else None, view_label=target_view.to(device) if cfg.MODEL.SIE_VIEW else None)
                if feat_norm:
                    gallery_feat = F.normalize(gallery_feat, p=2, dim=1)

                # 存储图库结果
                gallery_features.append(gallery_feat.cpu())
                gallery_pids.extend(pid)
                gallery_camids.extend(camid)

        processed_samples += img.shape[0]
        if processed_samples % 1000 == 0:
            logger.info(f"Processed {processed_samples}/{len(val_loader.dataset)} samples...")

    # --- 循环结束，进行评估 ---
    end_time = time.time()
    logger.info(f"Feature extraction finished in {end_time - start_time:.2f} seconds.")

    if not query_features:
         logger.error("No query features were generated (possibly due to TTPT being disabled or errors). Cannot compute metrics.")
         return 0.0, 0.0

    qf = torch.cat(query_features, dim=0)
    gf = torch.cat(gallery_features, dim=0)
    q_pids = np.asarray(query_pids)
    g_pids = np.asarray(gallery_pids)
    q_camids = np.asarray(query_camids)
    g_camids = np.asarray(gallery_camids)

    logger.info(f"Query features shape: {qf.shape}")
    logger.info(f"Gallery features shape: {gf.shape}")

    # --- 计算图文相似度矩阵 ---
    gallery_feat_dim = model_without_ddp.in_planes_proj
    gf_proj = gf[:,-gallery_feat_dim:]
    if feat_norm:
         gf_proj = F.normalize(gf_proj, p=2, dim=1)

    logger.info(f"Comparing Query Text Features ({qf.shape}) with Gallery Image Projection Features ({gf_proj.shape})")
    similarity_matrix = torch.matmul(qf, gf_proj.t())
    distmat = 1 - similarity_matrix
    distmat = distmat.cpu().numpy()

    # --- 计算 mAP 和 CMC (改编自 R1_mAP_eval.compute) ---
    logger.info("Computing metrics from similarity matrix...")
    cmc = np.zeros(len(g_pids))
    ap = np.zeros(len(q_pids))
    max_rank = 50

    for q_idx in range(len(q_pids)):
        # 获取当前查询的距离、pid、camid
        dist = distmat[q_idx]
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # 移除同一摄像头下的同一 ID 样本 (ReID 标准做法)
        order = np.argsort(dist)
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        ordered_dist = dist[order][keep]
        ordered_g_pids = g_pids[order][keep]
        ordered_g_camids = g_camids[order][keep]

        # 计算 CMC
        matches = (ordered_g_pids == q_pid).astype(np.int32)
        k = np.where(matches == 1)[0]
        if len(k) == 0:
            continue

        cmc_tmp = np.zeros(len(g_pids))
        cmc_tmp[k[0]:] = 1
        cmc += cmc_tmp[:len(cmc)]

        # 计算 AP (Average Precision)
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * matches
        ap[q_idx] = tmp_cmc.sum() / num_rel

    # 计算最终指标
    mAP = ap.mean()
    all_cmc = cmc / len(q_pids)

    logger.info("Validation Results (TTPT CLIP-style)")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        if r <= max_rank:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, all_cmc[r - 1]))
        else:
             logger.info(f"Rank-{r} exceeds max_rank ({max_rank})")


    # 返回 Rank-1 和 Rank-5
    rank1 = all_cmc[0] if len(all_cmc) > 0 else 0.0
    rank5 = all_cmc[4] if len(all_cmc) > 4 else 0.0
    logger.info(f"Returning Rank-1: {rank1:.1%}, Rank-5: {rank5:.1%}")

    torch.cuda.empty_cache()

    return rank1, rank5

# --- New Function for Option A ---
def do_inference_ttpt_option_a(cfg,
                               model,
                               val_loader,
                               num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test_ttpt_option_a") # Use a distinct logger
    logger.info("Enter inferencing with TTA only (Option A - Image Feature Evaluation)") # Modified log

    # --- Config Parameters ---
    tta_enabled = cfg.TEST.get('TTA_ENABLED', True)
    # ttpt_enabled = cfg.TEST.TTPT.get('ENABLED', True) # TTPT settings no longer needed for execution
    # ttpt_lr = cfg.TEST.TTPT.get('LR', 0.001)
    # ttpt_steps = cfg.TEST.TTPT.get('STEPS', 5)
    # ttpt_temperature = cfg.TEST.TTPT.get('TEMPERATURE', 0.07)
    feat_norm = cfg.TEST.FEAT_NORM

    # Log settings
    if tta_enabled:
        logger.info("Test Time Augmentation (TTA) enabled.")
    # if ttpt_enabled:
    #     logger.info(f"Test Time Prompt Tuning (TTPT) optimization enabled: LR={ttpt_lr}, Steps={ttpt_steps}, Temp={ttpt_temperature}")
    #     logger.info("Note: TTPT optimized text features are NOT directly used for matching in Option A.")
    # else:
    #      logger.info("TTPT optimization disabled.")
    logger.info("TTPT optimization part is disabled for this run.") # Added log

    # --- Model and Evaluator Setup ---
    if device:
        if torch.cuda.device_count() > 1:
            logger.info(f'Using {torch.cuda.device_count()} GPUs for inference')
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        model.to(device)
    else:
        model_without_ddp = model

    # Use the standard R1_mAP_eval for image-image comparison
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    model.eval()

    # --- Get TTPT Modules (Needed only if TTPT optimization is enabled) ---
    # Commenting out the TTPT module loading as it's not needed
    # prompt_learner = None
    # text_encoder = None
    # tokenized_prompts = None
    # all_class_labels = None
    # if ttpt_enabled:
    #     try:
    #         prompt_learner = model_without_ddp.prompt_learner
    #         text_encoder = model_without_ddp.text_encoder
    #         tokenized_prompts = prompt_learner.tokenized_prompts.to(device)
    #         num_classes = prompt_learner.num_class
    #         all_class_labels = torch.arange(num_classes, device=device)
    #         logger.info("Successfully accessed PromptLearner and TextEncoder for TTPT optimization.")
    #     except AttributeError as e:
    #         logger.error(f"Failed to get modules needed for TTPT optimization: {e}. Disabling TTPT optimization part.")
    #         ttpt_enabled = False # Disable TTPT if modules aren't found

    # --- Feature Extraction Loop ---
    logger.info("Starting feature extraction...")
    start_time = time.time()
    processed_samples = 0

    for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
        img = img.to(device)
        # Keep pid, camid as tuples, R1_mAP_eval handles them

        is_query = (processed_samples < num_query)

        with torch.no_grad(): # Outer no_grad for TTA and Gallery feature extraction
            if is_query:
                # --- 1. TTA & Feature Aggregation ---
                feat_list = []

                # a) Original Image Feature
                feat_orig = model(x=img, cam_label=camids.to(device) if cfg.MODEL.SIE_CAMERA else None, view_label=target_view.to(device) if cfg.MODEL.SIE_VIEW else None)
                feat_list.append(feat_orig)
                # logger.debug(f"Query {processed_samples}: Original feature extracted.") # Add debug log if needed

                # b) Flipped Image Feature
                if tta_enabled:
                    img_flipped = torch.flip(img, [3])
                    feat_flipped = model(x=img_flipped, cam_label=camids.to(device) if cfg.MODEL.SIE_CAMERA else None, view_label=target_view.to(device) if cfg.MODEL.SIE_VIEW else None)
                    feat_list.append(feat_flipped)
                    # logger.debug(f"Query {processed_samples}: Flipped feature extracted.") # Add debug log if needed

                # --- ADD Pseudo Modalities ---
                if tta_enabled: # Only add pseudo modalities if TTA is generally enabled
                    try:
                        # c) Pseudo-IR Feature (Simulated from RGB by channel averaging)
                        # Assumes img is a normalized Bx3xHxW tensor
                        img_gray = img.mean(dim=1, keepdim=True) # Bx1xHxW
                        pseudo_ir = img_gray.repeat(1, 3, 1, 1) # Bx3xHxW
                        feat_pseudo_ir = model(x=pseudo_ir, cam_label=camids.to(device) if cfg.MODEL.SIE_CAMERA else None, view_label=target_view.to(device) if cfg.MODEL.SIE_VIEW else None)
                        feat_list.append(feat_pseudo_ir)
                        # logger.debug(f"Query {processed_samples}: Pseudo-IR feature extracted.") # Add debug log if needed

                        # d) Pseudo-RGB Feature (Simulated from IR by channel replication)
                        # This is a simple simulation. We take the first channel and repeat it.
                        pseudo_rgb = img[:, 0:1, :, :].repeat(1, 3, 1, 1) # Take first channel, repeat 3 times
                        feat_pseudo_rgb = model(x=pseudo_rgb, cam_label=camids.to(device) if cfg.MODEL.SIE_CAMERA else None, view_label=target_view.to(device) if cfg.MODEL.SIE_VIEW else None)
                        feat_list.append(feat_pseudo_rgb)
                        # logger.debug(f"Query {processed_samples}: Pseudo-RGB feature extracted.") # Add debug log if needed

                    except Exception as e:
                         logger.warning(f"Query {processed_samples}: Failed to generate pseudo-modality features: {e}")
                # --- End Pseudo Modalities ---

                # Aggregate features (average) - Now includes original, flipped, pseudo-IR, pseudo-RGB
                if len(feat_list) > 0:
                    img_feat_agg = torch.stack(feat_list, dim=0).mean(dim=0)
                    if feat_norm:
                        img_feat_agg = F.normalize(img_feat_agg, p=2, dim=1)
                else:
                    logger.warning(f"Query {processed_samples}: No features generated in TTA list. Skipping.")
                    processed_samples += img.shape[0] # Ensure progress
                    continue # Skip to next iteration if no features

                # --- 2. TTPT Optimization (Indirect Role) --- (Commented out)
                # ... (TTPT code remains commented out) ...

                # --- 3. Update Evaluator ---
                # Use the AGGREGATED IMAGE FEATURE (from potentially 4 views) for evaluation
                evaluator.update((img_feat_agg, pid, camid)) # Pass tuple pid/camid

            # --- Gallery Processing ---
            else:
                # Extract standard gallery image feature
                gallery_feat = model(x=img, cam_label=camids.to(device) if cfg.MODEL.SIE_CAMERA else None, view_label=target_view.to(device) if cfg.MODEL.SIE_VIEW else None)
                if feat_norm:
                    gallery_feat = F.normalize(gallery_feat, p=2, dim=1)
                # Update evaluator with gallery image feature
                evaluator.update((gallery_feat, pid, camid)) # Pass tuple pid/camid

        processed_samples += img.shape[0]
        if processed_samples % 1000 == 0:
            logger.info(f"Processed {processed_samples}/{len(val_loader.dataset)} samples...")

    # --- Evaluation ---
    end_time = time.time()
    logger.info(f"Feature extraction finished in {end_time - start_time:.2f} seconds.")

    # Compute metrics using the standard evaluator
    cmc, mAP, _, _, _, _, _ = evaluator.compute() # This computes image-image metrics

    logger.info("Validation Results (TTPT Option A - Image Features)")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        # Use max_rank from evaluator if available, otherwise use a default like 50
        eval_max_rank = getattr(evaluator, 'max_rank', 50)
        if r <= eval_max_rank and r <= len(cmc):
             logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        else:
            logger.info(f"Rank-{r} exceeds max_rank ({eval_max_rank}) or CMC length ({len(cmc)}) ")

    # Return metrics
    rank1 = cmc[0] if len(cmc) > 0 else 0.0
    rank5 = cmc[4] if len(cmc) > 4 else 0.0
    logger.info(f"Returning Rank-1: {rank1:.1%}, Rank-5: {rank5:.1%}")

    torch.cuda.empty_cache()
    return rank1, rank5