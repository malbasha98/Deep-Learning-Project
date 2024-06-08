import argparse
import yaml
from train_single_model import train


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Train a model using a YAML configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Use the configuration to train the model
    train(args.config)


if __name__ == "__main__":
    main()
