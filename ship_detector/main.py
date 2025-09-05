from ship_detector.scripts.train_vit import train_vit_model
from ship_detector.scripts.utils import get_args, get_task
from ship_detector.scripts.train_unet import train_unet_model
from ship_detector.scripts.inference_pipeline import run_inference


def main():
    args = get_args()
    task = get_task(args)
    match task:
        case "train_vit":
            train_vit_model(
                config_path=args.config,
                output_dir=args.output_dir,
            )

        case "train_unet":
            train_unet_model(
                config_path=args.config,
                output_dir=args.output_dir,
            )

        case "inference":
            run_inference(
                inference_cfg_path=args.config,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
