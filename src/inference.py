import argparse
import logging
import os

import yaml
from data.make_dataset import getInferenceTFData
from models.pali import PaLI
from utils.infer_helper import (
    decode_output,
    get_infer_step,
    hardware_setting,
    load_checkpoint,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def main(args):
    # Testcase
    assert os.path.exists(
        args.config_path
    ), f"Config file {args.config_path} does not exist!"
    # Hardware setting
    hardware_setting(args)
    # Load the config file
    cfg = yaml.safe_load(open(args.config_path))

    # Init model
    model = PaLI(cfg)

    if not os.path.exists(args.ckpt):
        raise ValueError(f"Checkpoint file {args.ckpt} does not exist!")
    LOGGER.info(f"Loading PaLI checkpoint from {args.ckpt}")
    params = load_checkpoint(args.ckpt)

    LOGGER.info(
        "Preprocessing image {} and question {}".format(
            args.input_image, args.input_question
        )
    )
    hpparams = cfg["hyperparams"]
    dataloader, questions = getInferenceTFData(
        args.input_image, args.input_question, batch_size=1,
        encoder_input_tokens=hpparams['encoder_input_tokens'],
        decoder_target_tokens=hpparams['decoder_target_tokens'],
    )

    infer_step_fn = get_infer_step(cfg, model)

    for index, inputs in enumerate(dataloader):
        LOGGER.info("Sample {}".format(index))
        pred, score = infer_step_fn(params, inputs)
        for i in range(pred.shape[0]):
            # 1 here is the batch size
            LOGGER.info("Question: {}".format(questions[index * 1 + i]))
            LOGGER.info(
                "Predicted answer: {} with score {}".format(
                    decode_output(pred[i]), score[i]
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaLI training script")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default_config.yml",
        help="Path to the config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id")
    parser.add_argument(
        "--input_image",
        type=str,
        default="/kaggle/input/coco-textcaps-textvqa/tran_val_dataset/images/0000599864fd15b3.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--input_question",
        type=str,
        default="Generate the caption for the photo",
        help="Input question",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/kaggle/working/pali-model/src/checkpoints/params/params_20000.npz",
        help="Path to the checkpoint. (.npz)",
    )
    args = parser.parse_args()
    main(args)