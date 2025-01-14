#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-01 15:19:33
# Author: Luzhou Peng (彭路洲) & Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-09 09:47:28
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


'''
This project is based on the "nasbench_keras" repository by lienching 
(https://github.com/lienching/nasbench_keras), with modifications for
converting nasbench models to Younger instances.
'''

import os
import json
import tqdm
import click
import pathlib
import multiprocessing
import tensorflow as tf

from typing import Any

from tf2onnx import tf_loader, optimizer
from tf2onnx.tfonnx import process_tf_graph
from google.protobuf import text_format
from nasbench_keras import ModelSpec, build_keras_model

from younger.datasets.modules import Instance
from younger.commons.io import create_dir, load_json
from younger.commons.logging import set_logger, use_logger, logger

from younger.datasets.utils.convertors.tensorflow2onnx import main_export

def safe_tflite_export(tflite_model_path: pathlib.Path, onnx_model_filepath: pathlib.Path):
    main_export(tflite_model_path, onnx_model_filepath, model_type='keras')

       
def loads_json(serialized_object: str, cls: json.JSONDecoder | None = None) -> object:
    serializable_object = json.loads(serialized_object, cls=cls)
    return serializable_object


def saves_json(serializable_object: object, cls: json.JSONEncoder | None = None) -> str:
    serialized_object = json.dumps(serializable_object, sort_keys=True, cls=cls)
    return serialized_object

def save_protobuf(path, message, as_text=False):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(message))
    else:
        with open(path, "wb") as f:
            f.write(message.SerializeToString())


def convert_pipeline(params):
    # Adjacency matrix and nuberically-coded layer list
    model_info, cvt_cache_dirpath, ofc_cache_dirpath, instances_dirpath, opset, config = params
    model_id = model_info['id']
    matrix = model_info['matrix']
    labels = model_info['labels']

    keras_model_filepath = ofc_cache_dirpath.joinpath(f'{model_id}.keras')
    onnx_model_filepath = cvt_cache_dirpath.joinpath(f'{model_id}.onnx')

    # Transfer numerically-coded operations to layers (check base_ops.py)
    labels = (['input'] + [config['available_ops'][l] for l in labels[1:-1]] + ['output'])
    try:
        # Module graph
        spec = ModelSpec(matrix, labels, data_format='channels_first')

        # Create module
        features = tf.keras.layers.Input((3,224,224), 1)
        net_outputs = build_keras_model(spec, features, labels, config)
        net = tf.keras.Model(inputs=features, outputs=net_outputs)

        # Save the module
        net.save(keras_model_filepath)

        # Convert the module to ONNX
        onnx_model = safe_tflite_export(keras_model_filepath, onnx_model_filepath)

        instance = Instance(
            model=onnx_model,
            labels=dict(
                model_source='NAS-Bench-101',
                model_name=model_id,
                onnx_model_filename=onnx_model_filepath.name,
                download=None,
                like=None,
                tag=None,
                readme=None,
                annotations=None
            )
        )

        # Convert the ONNX model to YLIR instance
        instance.save(instances_dirpath)
        keras_model_filepath.unlink(missing_ok=True)
        # onnx_model_filepath.unlink(missing_ok=True)
        return True, model_id
    except Exception as exception:
        print(exception)
        return False, model_id


def get_convert_status_and_last_handled_model_id(sts_cache_dirpath: pathlib.Path, start_index: int, end_index: int) -> tuple[list[dict[str, dict[int, Any]]], str | None]:
    convert_status: dict[str, dict[int, Any]] = list()
    status_filepath = sts_cache_dirpath.joinpath(f's{start_index}_e{end_index}.sts')
    if status_filepath.is_file():
        with open(status_filepath, 'r') as status_file:
            convert_status: dict[str, dict[int, Any]] = [loads_json(line.strip()) for line in status_file]

    last_handled_filepath = sts_cache_dirpath.joinpath(f's{start_index}_e{end_index}_last_handled.sts')
    if last_handled_filepath.is_file():
        with open(last_handled_filepath, 'r') as last_handled_file:
            model_id = last_handled_file.read().strip()
        last_handled_model_id = model_id
    else:
        last_handled_model_id = None
    return convert_status, last_handled_model_id


def set_convert_status_last_handled_model_id(sts_cache_dirpath: pathlib.Path, start_index: int, end_index: int, convert_status: dict[str, dict[str, Any]], model_id: str):
    convert_status_filepath = sts_cache_dirpath.joinpath(f's{start_index}_e{end_index}.sts')
    with open(convert_status_filepath, 'a') as convert_status_file:
        convert_status_file.write(f'{saves_json((model_id, convert_status))}\n')

    last_handled_filepath = sts_cache_dirpath.joinpath(f's{start_index}_e{end_index}_last_handled.sts')
    with open(last_handled_filepath, 'w') as last_handled_file:
        last_handled_file.write(f'{model_id}\n')


@click.command()
@click.option('--model-infos-filepath', required=True,  type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), help='The filepath specifies the address of the Model Infos file, which is obtained using the command: `younger logics ir create onnx retrieve huggingface --mode Model_Infos ...`.')
@click.option('--save-dirpath',         required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--cache-dirpath',        required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='Cache directory, where data is volatile.')
@click.option('--opset',                required=True,  type=int, help='Used to indicate which opset version the model needs to be converted to in the ONNX format.')
@click.option('--start-index',          required=False, type=int, default=None, help='Used to indicate the position of the model in the model_infos list where the conversion starts.')
@click.option('--end-index',            required=False, type=int, default=None, help='Used to indicate the position of the model in the model_infos list where the conversion ends (the index are excluded).')
@click.option('--worker-number',        required=False, type=int, default=1, help='Used to indicate how many processes are concurrently performing the model conversion tasks.')
def main(
    model_infos_filepath,
    save_dirpath, cache_dirpath,
    opset,
    start_index, end_index,
    worker_number,
):
    set_logger('NAS-Bench-101_Convert', mode='console', level='INFO')
    use_logger('NAS-Bench-101_Convert')

    model_infos = load_json(model_infos_filepath)
    start_index = start_index or 0
    end_index = end_index or len(model_infos)
    model_infos = model_infos[start_index:end_index]

    # Instances
    # create_dir(instances_dirpath)

    # Official
    ofc_cache_dirpath = cache_dirpath.joinpath(f'Cache-NB101Ofc')
    create_dir(ofc_cache_dirpath)

    # Convert
    cvt_cache_dirpath = cache_dirpath.joinpath(f'Cache-NB101Cvt')
    create_dir(cvt_cache_dirpath)

    # Status
    sts_cache_dirpath = cache_dirpath.joinpath(f'Cache-NB101Sts')
    create_dir(sts_cache_dirpath)

    config = {
        'available_ops' : ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
        'stem_filter_size' : 128,
        'data_format' : 'channels_first',
        'num_stacks' : 3,
        'num_modules_per_stack' : 2,
        'num_labels' : 1000
    }

    convert_status, last_handled_model_id = get_convert_status_and_last_handled_model_id(sts_cache_dirpath, start_index, end_index)

    logger.info(f'Packaging Parameters For Conversion ...')
    params = list()
    with tqdm.tqdm(total=len(model_infos), desc='Packaging Params') as progress_bar:
        for model_info in model_infos:
            model_id = model_info['id']
            if last_handled_model_id is not None:
                if model_id == last_handled_model_id:
                    last_handled_model_id = None
                progress_bar.set_description(f'Converted, Skip - {model_id}')
                progress_bar.update(1)
                continue
            params.append((model_info, cvt_cache_dirpath, ofc_cache_dirpath, save_dirpath.joinpath(f'{model_id}'), opset, config))
            progress_bar.update(1)

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(params), desc='Converting Models') as progress_bar:
            for index, (status, model_id) in enumerate(pool.imap_unordered(convert_pipeline, params), start=1):
                set_convert_status_last_handled_model_id(sts_cache_dirpath, start_index, end_index, status, model_id)
                progress_bar.set_description(f'Converting - {model_id}')
                progress_bar.update(1)


if __name__ == '__main__':
    main()
