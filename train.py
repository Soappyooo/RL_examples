import yaml
import argparse
import os
import logging
import time


def get_current_path():
    return os.path.abspath(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config/curling/ppo.yml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Ensure the checkpoint directory exists
    os.makedirs(config["training"]["checkpoint_path"], exist_ok=True)
    os.makedirs(config["training"]["log_path"], exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config["training"]["log_path"], f"{time.strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler(),
        ],
    )
    logger.info(f"Current script path: {get_current_path()}")

    training_mode = config["training"].get("mode", "off-policy")
    if training_mode not in ["on-policy", "off-policy"]:
        raise ValueError(f"Invalid training mode: {training_mode}. Choose 'on-policy' or 'off-policy'.")

    if training_mode == "on-policy":
        from utils.trainer import train_on_policy

        train_on_policy(config, logger)
    elif training_mode == "off-policy":
        from utils.trainer import train_off_policy

        train_off_policy(config, logger)


if __name__ == "__main__":
    main()
