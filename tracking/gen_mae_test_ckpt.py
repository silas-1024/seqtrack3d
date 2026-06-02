import argparse
import os
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser("Generate a SeqTrack test checkpoint initialized from MAE pretrain")
    parser.add_argument("--yaml_name", type=str, default="seqtrack_b384_3d")
    parser.add_argument(
        "--mae_path",
        type=str,
        default="/home/silas/tracking/algorithm/seqtrack_3d/pretrained_models/mae_pretrain_vit_base.pth",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        default="/home/silas/tracking/algorithm/seqtrack_3d/output/checkpoints/train/seqtrack/seqtrack_b384_3d/SEQTRACK_ep0060.pth.tar",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    from lib.config.seqtrack.config import cfg, update_config_from_file
    from lib.models.seqtrack import build_seqtrack
    from lib.models.seqtrack import vit as vit_module
    from lib.models.seqtrack import vit_3d as vit_3d_module

    yaml_file = repo_root / "experiments" / "seqtrack" / f"{args.yaml_name}.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_file}")
    if not os.path.isfile(args.mae_path):
        raise FileNotFoundError(f"MAE checkpoint not found: {args.mae_path}")

    update_config_from_file(str(yaml_file))

    # Build model without remote downloading, then load local MAE pretrain manually.
    encoder_type = cfg.MODEL.ENCODER.TYPE
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = "scratch"

    model = build_seqtrack(cfg)

    local_pretrain_cfg = {
        "url": f"file://{args.mae_path}",
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        "num_classes": 1000,
    }

    if "vit_3d" in encoder_type.lower():
        vit_3d_module.load_pretrained(
            model.encoder.body,
            pretrain_type="mae",
            cfg=local_pretrain_cfg,
            num_classes=model.encoder.body.num_classes,
            in_chans=3,
            strict=False,
        )
    else:
        vit_module.load_pretrained(
            model.encoder.body,
            pretrain_type="mae",
            cfg=local_pretrain_cfg,
            num_classes=model.encoder.body.num_classes,
            in_chans=3,
            strict=False,
        )

    out_path = Path(args.output_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": model.state_dict(), "net_type": "SEQTRACK"}, str(out_path))
    print(f"Saved checkpoint: {out_path}")


if __name__ == "__main__":
    main()
