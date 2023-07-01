import argparse
import logging
import os

import gradio as gr
import yaml
from src.data.make_dataset import getInferenceTFData
from src.models.pali import PaLI
from src.utils.infer_helper import (
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
    
    infer_step_fn = get_infer_step(cfg, model)
    
    def inference(input_image, input_question, batch_size=1):
        dataloader, questions = getInferenceTFData(
            input_image, input_question, batch_size
        )
        results = []
        for index, inputs in enumerate(dataloader):
            LOGGER.info("Sample {}".format(index))
            pred, score = infer_step_fn(params, inputs)
            for i in range(pred.shape[0]):
                # 1 here is the batch size
                LOGGER.info("Question: {}".format(questions[index * 1 + i]))
                result = decode_output(pred[i])
                LOGGER.info(
                    "Predicted answer: {} with score {}".format(
                        result, score[i]
                    )
                )
        return str(result[0])
    
    def clear_inputs():
        return None, None, None
    
    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown(
                """
                <p align='center' style='font-size: 25px;'>PaLI Model (Pathways Language and Image Model)</p>
                """
                
            )
        with gr.Row():
            input_image = gr.Image(source='upload', type="filepath")
        with gr.Row():
            input_question = gr.Textbox(label="Question")
        with gr.Row():
            run_button = gr.Button(label="Run")
            with gr.Column():
                clear_button = gr.Button("Clear")
        with gr.Row():
            output_answer = gr.Textbox(label="Answer", interactive=True)
        # Example image
        gr.Examples(
            examples=[
                "https://farm3.staticflickr.com/8673/16416250538_594bcb48d2_o.jpg",
                "https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg",
                "https://farm7.staticflickr.com/2092/2209967213_d34f6cbd17_o.jpg",
                "https://c1.staticflickr.com/5/4147/5209877862_a1de7ee298_o.jpg",
                "https://farm7.staticflickr.com/7344/10624473206_2f93e40bef_o.jpg",
                "https://c2.staticflickr.com/6/5181/5660491676_4844ef43ab_o.jpg",
                "https://farm2.staticflickr.com/3876/14701969871_397bbd5ae6_o.jpg"
            ],
            inputs=input_image,
            outputs=None,
            fn=None,
            cache_examples=False,
        )
        ips = [input_image, input_question]
        run_button.click(fn=inference, inputs=ips, outputs=output_answer)
        clear_button.click(fn=clear_inputs, inputs=None, outputs=[input_image, input_question, output_answer])
    block.launch(server_name='0.0.0.0', debug=True, share=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaLI training script")
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/configs/default_config.yml",
        help="Path to the config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="src/checkpoints/params/params_32000.npz",
        help="Path to the checkpoint. (.npz)",
    )
    args = parser.parse_args()
    main(args)