from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.seqtrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/seqtrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)
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
    print(f"  AFFINE_CACHE_ENABLE: {motion_cfg.get('AFFINE_CACHE_ENABLE', False) if motion_cfg else False}")
    print(f"  AFFINE_CACHE_ROOT: {motion_cfg.get('AFFINE_CACHE_ROOT', '') if motion_cfg else ''}")
    print("  online ORB/RANSAC: false")
    print(f"  motion input: {delta_type}_delta (normalized xywh difference x 128)")
    print(f"  decoder V-Gating enabled: {v_gating_enabled}")
    print(f"  affine residual compensation: {delta_type == 'residual'}")
    print("  coordinate prior: disabled")

    params.yaml_name = yaml_name
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join("./checkpoints/train/seqtrack/%s/SEQTRACK_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
