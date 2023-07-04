import inspect
import os

import numpy as np
import pandas as pd
import seqio
import t5
import tensorflow as tf
from PIL import Image
from seqio import FeatureConverter, non_padding_position, utils


# Ref: https://github.com/google/seqio/blob/main/seqio/feature_converters.py#:~:text=class%20EncDecFeatureConverter(,return%20model_feature_lengths
# Modify the class EncDecFeatureConverter to support images retrieval
class EncDecFeatureConverter(FeatureConverter):
    """Feature converter for an encoder-decoder architecture.
    
    The input dataset has "inputs" and "targets" field. These will be converted
    to a subset of standard features.
    
    To use packing, pass pack = True argument to the FeatureConverter's
    constructor. When packing is done, two additional fields are added for each of
    "inputs" and "targets" fields.
    
    Example for a packed dataset:
    
    The input dataset has two examples each with "inputs" and "targets".
    
    ds = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1]},
            {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1]}]
    
    task_feature_lengths = {"inputs": 10, "targets": 7}
    
    First, the `inputs` are packed together, padded to length 10 and assigned to
    "encoder_input_tokens" field. The `targets` are processed similarly.
    
    The "*_segment_id" fields are generated from the packing operation. For the
    explanation of these fields, see the module docstring.
    
    The "decoder_loss_weights" is a binary mask indicating where non-padding
    positions are, i.e., value of 1 indicates non-padding and 0 for padding. This
    class assumes that the loss is taken only on the decoder side.
    
    converted_ds = [{
        "encoder_input_tokens": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
            "encoder_segment_ids": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
            "encoder_positions": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 0, 4, 0, 0],
        "decoder_loss_weights": [1, 1, 1, 1, 1, 0, 0],
            "decoder_segment_ids": [1, 1, 1, 2, 2, 0, 0],
            "decoder_positions": [0, 1, 2, 0, 1, 0, 0],
    }]
    
    Note that two examples are packed together into one example.
    """
    
    TASK_FEATURES = {
        "images": FeatureConverter.FeatureSpec(dtype=tf.string),
        "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
    }
    MODEL_FEATURES = {
        "images": FeatureConverter.FeatureSpec(dtype=tf.string),
        "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
        "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
    }
    PACKING_FEATURE_DTYPES = {
        "encoder_segment_ids": tf.int32,
        "decoder_segment_ids": tf.int32,
        "encoder_positions": tf.int32,
        "decoder_positions": tf.int32,
    }
    
    def _convert_example(self, features):
        """Convert a seq2seq example into an example with model features."""
        # targets_segment_id is present only for a packed dataset.
        decoder_input_tokens = utils.make_autoregressive_inputs(
            features["targets"],
            sequence_id=features.get("targets_segment_ids", None),
            bos_id=self.bos_id,
        )
        
        d = {
            "images": features["images"],
            "encoder_input_tokens": features["inputs"],
            "decoder_target_tokens": features["targets"],
            "decoder_input_tokens": decoder_input_tokens,
            # Loss is computed for all but the padding positions.
            "decoder_loss_weights": non_padding_position(features["targets"]),
        }
        
        if self.pack:
            d["images"] = features["images"]
            d["encoder_segment_ids"] = features["inputs_segment_ids"]
            d["decoder_segment_ids"] = features["targets_segment_ids"]
            d["encoder_positions"] = features["inputs_positions"]
            d["decoder_positions"] = features["targets_positions"]
        
        return d
    
    def _convert_features(self, ds, task_feature_lengths):
        """Convert the dataset to be fed to the encoder-decoder model.
        
        The conversion process involves two steps
        
        1. Each feature in the `task_feature_lengths` is trimmed/padded and
        optionally packed depending on the value of self.pack.
        2. "inputs" fields are mapped to the encoder input and "targets" are mapped
        to decoder input (after being shifted) and target.
        
        All the keys in the `task_feature_lengths` should be present in the input
        dataset, which may contain some extra features that are not in the
        `task_feature_lengths`. They will not be included in the output dataset.
        One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
        fields.
        
        Args:
        ds: an input tf.data.Dataset to be converted.
        task_feature_lengths: a mapping from feature to its length.
        
        Returns:
        ds: the converted dataset.
        """
        ds = self._pack_or_pad(ds, task_feature_lengths)
        return ds.map(
            self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    
    def get_model_feature_lengths(self, task_feature_lengths):
        """Define the length relationship between input and output features."""
        images_length = task_feature_lengths["images"]
        encoder_length = task_feature_lengths["inputs"]
        decoder_length = task_feature_lengths["targets"]
        
        model_feature_lengths = {
            "images": images_length,
            "encoder_input_tokens": encoder_length,
            "decoder_target_tokens": decoder_length,
            "decoder_input_tokens": decoder_length,
            "decoder_loss_weights": decoder_length,
        }
        if self.pack:
            model_feature_lengths["images"] = images_length
            model_feature_lengths["encoder_segment_ids"] = encoder_length
            model_feature_lengths["decoder_segment_ids"] = decoder_length
            model_feature_lengths["encoder_positions"] = encoder_length
            model_feature_lengths["decoder_positions"] = decoder_length
        
        return model_feature_lengths


def getTFDataGenerator(
    data_root: str,
    csv_file: str,
    batch_size: int = 64,
    resize: tuple = (224, 224),
    rescale: float = 1.0 / 255,
    encoder_input_tokens: int = 38,
    decoder_target_tokens: int = 18,
    add_eos: bool = True,
    shuffle: bool = True,
):
    """Create Tensorflow Data Generator with pre-processing for Visual Question Answering
    
    Args:
        data_root (str): Path to the data root directory
        csv_file (str): Path to the csv file
        batch_size (int, optional): return the pre-processed data in batches. Defaults to 64.
        resize (tuple, optional): resize the image to the given size. Defaults to (224, 224).
        rescale (float, optional): the image will be multiplied by the given value. Defaults to 1.0 / 255.
        encoder_input_tokens (int, optional): the maximum number of tokens for the encoder input. Defaults to 38.
        decoder_target_tokens (int, optional): the maximum number of tokens for the decoder target. Defaults to 18.
        add_eos (bool, optional): add end of sentence token to the decoder target. Defaults to True.
        shuffle (bool, optional): shuffle the dataset. Defaults to True.
    
    Returns:
        tf.data.Dataset: Tensorflow Data Generator in numpy format
        int: the number of samples in the dataset
    """
    "Init Tensorflow Data Generator"
    # Load csv file
    data = pd.read_csv(csv_file)
    data_images = data["image"]
    data_questions = data["question"]
    data_answers = data["answer"]
    # Create dictionary data
    merged_samples = {"images": [], "inputs": [], "targets": []}
    for i, q, a in zip(data_images, data_questions, data_answers):
        merged_samples["images"].append(os.path.join(data_root, i))
        merged_samples["inputs"].append(q)
        merged_samples["targets"].append(a)
    dataset = tf.data.Dataset.from_tensor_slices(merged_samples)
    
    "Tensorflow function for pre-processing"
    # Infinity dataset
    dataset = dataset.repeat()
    
    if shuffle:
        # Shuffle dataset, "I guess the value for buffer_size is about 1/2 of dataset for better shuffle"
        dataset = dataset.shuffle(buffer_size=len(data) // 2)
    
    # Prefetch dataset for faster training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Fix assert error in text_preprocessor create by seqio
    def _images_expand_dims(inputs_data):
        inputs_data["images"] = tf.expand_dims(inputs_data["images"], axis=0)
        return inputs_data
    
    dataset = dataset.map(_images_expand_dims, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Pre-process text: tokenize, append eos, map to vocab ids
    _text_preprocessing = [
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
    ]
    _text_features = {
        "inputs": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=add_eos
        ),
        "targets": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=add_eos
        ),
    }
    _text_feature_lengths = {
        "images": 1,
        "inputs": encoder_input_tokens,
        "targets": decoder_target_tokens,
    }
    for prep_fn in _text_preprocessing:
        fn_args = set(inspect.signature(prep_fn).parameters.keys())
        kwargs = {}
        if "sequence_length" in fn_args:
            kwargs["sequence_length"] = _text_feature_lengths
        if "output_features" in fn_args:
            kwargs["output_features"] = _text_features
        dataset = prep_fn(dataset, **kwargs)
    _text_features_converter_cls = EncDecFeatureConverter(pack=False)
    dataset = _text_features_converter_cls(
        dataset, task_feature_lengths=_text_feature_lengths
    )
    
    # Load image, resize, rescale
    def _image_PIL_preprocessing(path):
        img = Image.open(path.decode("utf-8")).resize(resize).convert("RGB")
        img = np.array(img, dtype="float32") * rescale
        return img
    
    def _image_preprocessing(inputs_data):
        img_path = inputs_data["images"]
        # Unsqueeze image path
        img_path = tf.squeeze(img_path, axis=0)
        img = tf.numpy_function(_image_PIL_preprocessing, [img_path], tf.float32)
        inputs_data["images"] = img
        return inputs_data
    
    dataset = dataset.map(_image_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Modify return value
    def _return_dataset(inputs_data):
        return {
            "images": inputs_data["images"],
            "encoder_input_tokens": inputs_data["encoder_input_tokens"],
            "decoder_input_tokens": inputs_data["decoder_input_tokens"],
            "decoder_target_tokens": inputs_data["decoder_target_tokens"],
        },  inputs_data["decoder_loss_weights"]
    
    dataset = dataset.map(_return_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add batch size
    dataset = dataset.batch(batch_size)
    
    return dataset.as_numpy_iterator(), len(data)


def getInferenceTFData(
    image_paths,
    questions,
    batch_size: int = 1,
    resize: tuple = (224, 224),
    rescale: float = 1.0 / 255,
    encoder_input_tokens: int = 38,
    decoder_target_tokens: int = 18,
    add_eos: bool = True,
):
    """A function to preprocess a batch of images and questions
    
    Args:
        image_path (str/list): path to the image or list of paths to the images
        question (str/list): path to the image or list of paths to the images
        batch_size (int, optional): return the pre-processed data in batches. Defaults to 64.
        resize (tuple, optional): resize the image to the given size. Defaults to (224, 224).
        rescale (float, optional): the image will be multiplied by the given value. Defaults to 1.0 / 255.
        encoder_input_tokens (int, optional): the maximum number of tokens for the encoder input. Defaults to 38.
        decoder_target_tokens (int, optional): the maximum number of tokens for the decoder target. Defaults to 18.
        add_eos (bool, optional): add end of sentence token to the decoder target. Defaults to True.
    
    Returns:
        tf.data.Dataset: Tensorflow Data Generator in numpy format
        int: the number of samples in the dataset
    """
    # Check if image_path is a string then convert it to list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if isinstance(questions, str):
        questions = [questions]
    assert len(image_paths) == len(
        questions
    ), "The number of images and questions must be equal"
    
    # Create dictionary data
    merged_samples = {"images": [], "inputs": [], "targets": []}
    for image_path, question in zip(image_paths, questions):
        merged_samples["images"].append(image_path)
        merged_samples["inputs"].append(question)
        merged_samples["targets"].append("net")
    
    dataset = tf.data.Dataset.from_tensor_slices(merged_samples)
    
    # Fix assert error in text_preprocessor create by seqio
    def _images_expand_dims(inputs_data):
        inputs_data["images"] = tf.expand_dims(inputs_data["images"], axis=0)
        return inputs_data
    
    dataset = dataset.map(_images_expand_dims, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Pre-process text: tokenize, append eos, map to vocab ids
    _text_preprocessing = [
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
    ]
    _text_features = {
        "inputs": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=add_eos
        ),
        "targets": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=add_eos
        ),
    }
    _text_feature_lengths = {
        "images": 1,
        "inputs": encoder_input_tokens,
        "targets": decoder_target_tokens,
    }
    for prep_fn in _text_preprocessing:
        fn_args = set(inspect.signature(prep_fn).parameters.keys())
        kwargs = {}
        if "sequence_length" in fn_args:
            kwargs["sequence_length"] = _text_feature_lengths
        if "output_features" in fn_args:
            kwargs["output_features"] = _text_features
        dataset = prep_fn(dataset, **kwargs)
    _text_features_converter_cls = EncDecFeatureConverter(pack=False)
    dataset = _text_features_converter_cls(
        dataset, task_feature_lengths=_text_feature_lengths
    )
    
    # Load image, resize, rescale
    def _image_PIL_preprocessing(path):
        img = Image.open(path.decode("utf-8")).resize(resize).convert("RGB")
        img = np.array(img, dtype="float32") * rescale
        return img
    
    def _image_preprocessing(inputs_data):
        img_path = inputs_data["images"]
        # Unsqueeze image path
        img_path = tf.squeeze(img_path, axis=0)
        img = tf.numpy_function(_image_PIL_preprocessing, [img_path], tf.float32)
        inputs_data["images"] = img
        return inputs_data
    
    dataset = dataset.map(_image_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Modify return value
    def _return_dataset(inputs_data):
        return {
            "images": inputs_data["images"],
            "encoder_input_tokens": inputs_data["encoder_input_tokens"],
            "decoder_input_tokens": inputs_data["decoder_input_tokens"],
            "decoder_target_tokens": inputs_data["decoder_target_tokens"],
        }
    
    dataset = dataset.map(_return_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add batch size
    dataset = dataset.batch(batch_size)
    
    return dataset.as_numpy_iterator(), questions