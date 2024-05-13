import argparse
import os
import yaml

from preparation import chem


def main(args, preprocess_config):
    os.makedirs("./lexicon", exist_ok=True)
    os.makedirs("./preprocessed_data", exist_ok=True)
    os.makedirs("./montreal-forced-aligner", exist_ok=True)

    if "Chem" in preprocess_config["dataset"]:
        if args.extract_lexicon:
            chem.extract_lexicon(preprocess_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        '--extract_nonen',
        help='extract non english charactor',
        action='store_true',
    )
    parser.add_argument(
        '--extract_lexicon',
        help='extract lexicon and build grapheme-phoneme dictionary',
        action='store_true',
    )
    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )

    main(args, preprocess_config)
