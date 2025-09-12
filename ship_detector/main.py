from ship_detector.scripts.trainer import train_vit_model, train_unet_model, train_sam_model, train_yolov8_model
from ship_detector.scripts.utils import get_args, get_task

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

        case "train_sam":
            train_sam_model(
                config_path=args.config,
                output_dir=args.output_dir,
            )
        
        case "train_yolo":
            train_yolov8_model(
                config_path=args.config,
                data_path=args.output_dir,
            )


if __name__ == "__main__":
    main()
