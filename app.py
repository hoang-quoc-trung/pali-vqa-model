import argparse
import logging
import os
import gradio as gr
from PIL import Image
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
    hpparams = cfg["hyperparams"]
    
    
    def inference(input_image, input_question, batch_size=1):
        if not input_image.endswith('.jpg'):
            im = Image.open(input_image)
            rgb_im = im.convert('RGB')
            input_image = input_image.replace('.png', '.jpg')
            rgb_im.save(str(input_image), "JPEG")
        dataloader, questions = getInferenceTFData(
            image_paths=input_image,
            questions=input_question,
            batch_size=batch_size,
            encoder_input_tokens=hpparams['encoder_input_tokens'],
            decoder_target_tokens=hpparams['decoder_target_tokens'],
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
                <p align='center' style='font-size: 25px;'>PaLI Model (Pathways Language and Image Model) </p>
                """
            )
        with gr.Row():
            input_image = gr.Image(source='upload', type="filepath", label="Image")
        with gr.Row():
            input_question = gr.Textbox(label="Question")
        with gr.Row():
            run_button = gr.Button(label="Run")
            with gr.Column():
                clear_button = gr.Button("Clear")
        with gr.Row():
            output_answer = gr.Textbox(label="Answer")

        gr.Examples(
            examples=[
                ["src/data/test_images/img_1.jpg", "How many people are in the picture?"],
                ["src/data/test_images/img_2.jpg", "What is the name of this plane?"],
                ["src/data/test_images/img_3.jpg", "What animal is grazing?"],
                ["src/data/test_images/img_4.jpg", "What color is the sofa?"],
                ["src/data/test_images/img_5.jpg", "What food is on the plate?"],
                ["src/data/test_images/img_6.jpg", "What animal is this?"],
                ["src/data/test_images/img_7.jpg", "What season is it?"],
                ["src/data/test_images/img_8.jpg", "Generate caption for the photo "],
            ],
            inputs=[input_image, input_question],
            outputs=None,
            fn=None,
            cache_examples=False,
            examples_per_page=4,
        )
        
        run_button.click(fn=inference, inputs=[input_image, input_question], outputs=output_answer)
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
        default="./src/checkpoints/params.npz",
        help="Path to the checkpoint. (.npz)",
    )
    args = parser.parse_args()
    main(args)