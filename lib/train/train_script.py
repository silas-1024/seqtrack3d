import os
# loss function related
from torch.nn import CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.seqtrack import build_seqtrack
# from lib.models.seqtrackv2 import build_seqtrackv2
# forward propagation related
from lib.train.actors import SeqTrackActor, SeqTrackV2Actor
# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for SeqTrack v1 and v2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file) #update cfg from experiments
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')
        motion_cfg = getattr(cfg.MODEL, "MOTION", None)
        motion_enabled = motion_cfg is not None \
            and motion_cfg.get("ENABLE", False) \
            and motion_cfg.get("ENABLE_MOTION_ENCODER", True)
        v_gating_enabled = motion_enabled \
            and motion_cfg.get("ENABLE_MOTION_GUIDED_ATTN", True)
        delta_type = motion_cfg.get("MOTION_DELTA_TYPE", "raw") if motion_cfg else "raw"
        experiment_name = "E2-residual-delta-RMP-VGate" \
            if delta_type == "residual" else "E1-raw-RMP-VGate"
        print(f"[{experiment_name}]")
        print(f"  motion enabled: {motion_enabled}")
        print(f"  MOTION_DELTA_TYPE: {delta_type}")
        print(f"  motion input: {delta_type}_delta (normalized xywh difference x 128)")
        print(f"  decoder V-Gating enabled: {v_gating_enabled}")
        print(f"  affine residual compensation: {delta_type == 'residual'}")
        print(f"  AFFINE_CACHE_ENABLE: {motion_cfg.get('AFFINE_CACHE_ENABLE', False) if motion_cfg else False}")
        print(f"  AFFINE_CACHE_ROOT: {motion_cfg.get('AFFINE_CACHE_ROOT', '') if motion_cfg else ''}")
        print(f"  AFFINE_CACHE_FALLBACK: {motion_cfg.get('AFFINE_CACHE_FALLBACK', 'identity') if motion_cfg else 'identity'}")
        print("  DataLoader online ORB/RANSAC: false")
        print("  coordinate prior: disabled\n")

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")


    # Create network
    if settings.script_name == "seqtrack":
        net = build_seqtrack(cfg)
    # elif settings.script_name == "seqtrackv2":
    #     net = build_seqtrackv2(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    # Loss functions and Actors
    if settings.script_name == "seqtrack":
        bins = cfg.MODEL.BINS
        weight = torch.ones(bins + 2)
        weight[bins] = 0.01
        weight[bins + 1] = 0.01
        objective = {'ce': CrossEntropyLoss(weight=weight)}
        loss_weight = {'ce': cfg.TRAIN.CE_WEIGHT}
        actor = SeqTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "seqtrackv2":
        bins = cfg.MODEL.BINS
        weight = torch.ones(bins + 2)
        weight[bins] = 0.01
        weight[bins + 1] = 0.01
        objective = {'ce': CrossEntropyLoss(weight=weight)}
        loss_weight = {'ce': cfg.TRAIN.CE_WEIGHT}
        actor = SeqTrackV2Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    

    # train process
    # trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, load_previous_ckpt=True,config_name = settings.config_name)
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, load_previous_ckpt=False,config_name = settings.config_name)
