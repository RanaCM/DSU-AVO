import argparse

import yaml

from preprocessor.preprocessor_chem import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    parser.add_argument("-g", "--generate_data_splits", action="store_true", help="generate from pre-assigned data splits")
    parser.add_argument("-c", "--copy_files", action="store_true", help="copy files for debug")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    if args.generate_data_splits:
        preprocessor.generate_data_splits()
    elif args.copy_files:
        preprocessor.copy_files()
    else:
        preprocessor.build_from_path()
