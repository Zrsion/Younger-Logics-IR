#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-26 14:24:10
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import re
import json
import tqdm
import pathlib
import multiprocessing

from typing import Literal

from huggingface_hub import HfFileSystem, login, hf_hub_download, snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError

from younger.commons.io import load_json, create_dir, delete_dir, get_human_readable_size_representation
from younger.commons.logging import logger

from younger_logics_ir.modules import Instance, LogicX, Implementation, Origin, Benchmark, Evaluation
from younger_logics_ir.converters import convert

from younger_logics_ir.dataset.utils import get_instance_dirname
from younger_logics_ir.scripts.commons.huggingface_utils import infer_model_size, clean_model_default_cache, clean_model_specify_cache, get_huggingface_model_readme, get_huggingface_model_card_data_from_readme, get_huggingface_model_info, get_huggingface_model_file_indicators
from younger.logics.ir.younger_logics_ir.scripts.commons.huggingface_annos import get_heuristic_annotations


def save_status(status_filepath: pathlib.Path, status: dict[str, str]):
    with open(status_filepath, 'a') as status_file:
        status = json.dumps(status)
        status_file.write(f'{status}\n')


def clean_all_cache(model_id: str, convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path):
    clean_model_default_cache(model_id)
    delete_dir(convert_cache_dirpath, only_clean=True)
    clean_model_specify_cache(model_id, huggingface_cache_dirpath)


def safe_optimum_export(model_id: str, convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path, device: str, flag_queue: multiprocessing.Queue):
    from optimum.exporters.onnx import main_export
    try:
        main_export(model_id, convert_cache_dirpath, device=device, cache_dir=huggingface_cache_dirpath, monolith=True, do_validation=False, trust_remote_code=True, no_post_process=True)
        flag_queue.put('success')
    except MemoryError as error:
        logger.error(f'Model ID = {model_id}: Skip! Maybe OOM - {error}')
        flag_queue.put('memory_error')
    except RepositoryNotFoundError as error:
        logger.error(f'Model ID = {model_id}: Skip! Maybe Deleted By Author - {error}')
        flag_queue.put('access_error')
    except Exception as error:
        logger.error(f'Model ID = {model_id}: Conversion Error - {error}')
        flag_queue.put('convert_error')


def convert_optimum(model_id: str, convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path, hf_file_system: HfFileSystem, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[str, list[pathlib.Path]]:
    assert device in {'cpu', 'cuda'}

    flag_queue = multiprocessing.Queue()
    subprocess = multiprocessing.Process(target=safe_optimum_export, args=(model_id, convert_cache_dirpath, huggingface_cache_dirpath, device, flag_queue))
    subprocess.start()
    subprocess.join()
    if flag_queue.empty():
        logger.warning(f'Export Process May Be Killed By System! Skip.')
        flag = 'system_kill'
    else:
        flag = flag_queue.get()

    onnx_model_filepaths: list[pathlib.Path] = list()
    for filepath in convert_cache_dirpath.iterdir():
        if filepath.suffix == '.onnx':
            onnx_model_filepaths.append(filepath)
    if len(onnx_model_filepaths) == 0:
        flag = 'convert_nothing'
    return flag, onnx_model_filepaths


def convert_onnx(model_id: str, convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path, hf_file_system: HfFileSystem, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[str, list[pathlib.Path]]:
    flag = 'success'
    onnx_model_filepaths: list[pathlib.Path] = list()
    remote_onnx_model_file_indicators = get_huggingface_model_file_indicators(model_id, ['.onnx'])
    for remote_onnx_model_dirpath, remote_onnx_model_filename in remote_onnx_model_file_indicators:
        try:
            onnx_model_filepath = hf_hub_download(model_id, os.path.join(remote_onnx_model_dirpath, remote_onnx_model_filename), cache_dir=huggingface_cache_dirpath)
        except Exception as error:
            logger.warning(f'Access Denied. Server Error - {error}')
            continue
        onnx_model_filepaths.append(pathlib.Path(onnx_model_filepath))
    if len(onnx_model_filepaths) == 0:
        flag = 'convert_nothing'
    return flag, onnx_model_filepaths


def safe_keras_export(keras_model_path: pathlib.Path, onnx_model_filepath: pathlib.Path, flag_queue: multiprocessing.Queue):
    from younger_logics_ir.scripts.commons.huggingface_miscs import tf2onnx_main_export
    if keras_model_path.is_dir():
        model_type = 'saved_model'
    if keras_model_path.is_file():
        model_type = 'keras'
    try:
        tf2onnx_main_export(keras_model_path, onnx_model_filepath, model_type=model_type)
        flag_queue.put('success')
    except Exception as error:
        flag_queue.put('convert_error')


def convert_keras(model_id: str, convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path, hf_file_system: HfFileSystem, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[str, list[pathlib.Path]]:
    overall_flag = 'success'
    onnx_model_filepath = convert_cache_dirpath.joinpath('model.onnx')

    remote_keras_model_paths = list()
    remote_keras_model_filepaths = list()
    for remote_keras_model_dirpath, remote_keras_model_filename in get_huggingface_model_file_indicators(model_id, ['.keras', '.hdf5', '.h5']):
        remote_keras_model_filepaths.append(os.path.join(remote_keras_model_dirpath, remote_keras_model_filename))
    remote_keras_model_dirpaths = list()
    for remote_keras_model_dirpath, remote_keras_model_filename in get_huggingface_model_file_indicators(model_id, ['.pbtxt', '.pb']):
        remote_keras_model_dirpaths.append(remote_keras_model_dirpath)
    remote_keras_model_paths = remote_keras_model_paths + remote_keras_model_filepaths + list(set(remote_keras_model_dirpaths))

    onnx_model_filepaths = list()
    if remote_keras_model_paths:
        keras_model_dirpath = snapshot_download(model_id, cache_dir=huggingface_cache_dirpath)
        keras_model_dirpath = pathlib.Path(keras_model_dirpath)
        for index, remote_keras_model_path in enumerate(remote_keras_model_paths):
            keras_model_path = keras_model_dirpath.joinpath(remote_keras_model_path)
            if keras_model_path.is_dir():
                onnx_model_filepath = convert_cache_dirpath.joinpath(f'from_pb_{index}.onnx')
            if keras_model_path.is_file():
                suffix_name = keras_model_path.suffix[1:]
                onnx_model_filepath = convert_cache_dirpath.joinpath(f'from_{suffix_name}_{index}.onnx')
            flag_queue = multiprocessing.Queue()
            subprocess = multiprocessing.Process(target=safe_keras_export, args=(keras_model_path, onnx_model_filepath, flag_queue))
            subprocess.start()
            subprocess.join()
            if flag_queue.empty():
                flag = 'system_kill'
            else:
                flag = flag_queue.get()
            logger.info(f'    ~ No.{index} Conversion Status: {flag} - Model Path: {keras_model_path}.')

            if onnx_model_filepath.is_file():
                onnx_model_filepaths.append(onnx_model_filepath)

    if len(onnx_model_filepaths) == 0:
        overall_flag = 'convert_nothing'
    return overall_flag, onnx_model_filepaths


def safe_tflite_export(tflite_model_path: pathlib.Path, onnx_model_filepath: pathlib.Path, flag_queue: multiprocessing.Queue):
    from younger_logics_ir.scripts.commons.huggingface_miscs import tf2onnx_main_export
    try:
        tf2onnx_main_export(tflite_model_path, onnx_model_filepath, model_type='tflite')
        flag_queue.put('success')
    except Exception as error:
        flag_queue.put('convert_error')


def convert_tflite(model_id: str, convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path, hf_file_system: HfFileSystem, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[str, list[pathlib.Path]]:
    overall_flag = 'success'
    onnx_model_filepaths: list[pathlib.Path] = list()
    remote_tflite_model_file_indicators = get_huggingface_model_file_indicators(model_id, ['.tflite'])
    for index, (remote_tflite_model_dirpath, remote_tflite_model_filename) in enumerate(remote_tflite_model_file_indicators):
        tflite_model_filepath = hf_hub_download(model_id, os.path.join(remote_tflite_model_dirpath, remote_tflite_model_filename), cache_dir=huggingface_cache_dirpath)
        tflite_model_filepath = pathlib.Path(tflite_model_filepath)
        onnx_model_filepath = convert_cache_dirpath.joinpath(remote_tflite_model_dirpath).joinpath(f'{os.path.splitext(remote_tflite_model_filename)[0]}.onnx')
        flag_queue = multiprocessing.Queue()
        subprocess = multiprocessing.Process(target=safe_tflite_export, args=(tflite_model_filepath, onnx_model_filepath, flag_queue))
        subprocess.start()
        subprocess.join()
        if flag_queue.empty():
            flag = 'system_kill'
        else:
            flag = flag_queue.get()
        logger.info(f'    ~ No.{index} Conversion Status: {flag} - Model Path: {tflite_model_filepath}.')

        if onnx_model_filepath.is_file():
            onnx_model_filepaths.append(onnx_model_filepath)

    if len(onnx_model_filepaths) == 0:
        overall_flag = 'convert_nothing'
    return overall_flag, onnx_model_filepaths


def main(
    model_ids_filepath: pathlib.Path,
    save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path,
    status_filepath: pathlib.Path | None = None,
    device: Literal['cpu', 'cuda'] = 'cpu',
    framework: Literal['optimum', 'onnx', 'keras', 'tflite'] = 'optimum',
    model_size_threshold: int | None = None,
    token: str | None = None,
):
    model_ids: set[str] = set(load_json(model_ids_filepath))

    assert framework in {'optimum', 'onnx', 'keras', 'tflite'}
    support_convert_method = dict(
        optimum = convert_optimum,
        onnx = convert_onnx,
        keras = convert_keras,
        tflite = convert_tflite,
    )
    save_dirpath = save_dirpath.joinpath(f'Instances-HuggingFace-{framework.title()}')

    if token is not None:
        logger.info(f'-> HuggingFace Token Provided. Now Logging In ...')
        login(token)
    else:
        logger.info(f'-> HuggingFace Token Not Provided. Now Accessing Without Token ...')

    hf_file_system = HfFileSystem()

    conversion_cache_dirpath = cache_dirpath.joinpath(f'ConversionCache-HuggingFace-{framework.title()}')
    create_dir(conversion_cache_dirpath)

    huggingface_cache_dirpath = cache_dirpath.joinpath('HuggingFaceCache')
    create_dir(huggingface_cache_dirpath)

    instance_dirpaths = list(save_dirpath.iterdir())
    with tqdm.tqdm(total=len(instance_dirpaths), desc='Checking Existing Converted Instances') as progress_bar:
        for index, instance_dirpath in enumerate(instance_dirpaths, start=1):
            instance = Instance()
            instance.load(instance_dirpath)
            assert len(instance.labels) == 1
            implementation = instance.labels[0]
            if implementation.origin.hub == 'HuggingFace':
                logger.info(f' . Converted. Skip Total {index} - {instance.labels["model_name"]}')
                model_ids = model_ids - {instance.labels['model_name']}

    if status_filepath.is_file():
        logger.info(f'-> Found Existing Status File')
        logger.info(f'-> Now Checking Status File ...')
        with open(status_filepath, 'r') as status_file:
            for index, line in enumerate(status_file, start=1):
                try:
                    status = json.loads(line)
                except:
                    logger.warning(f' . Skip No.{index}. Parse Error: Line in Status File: {line}')
                    continue

                if status['model_name'] not in model_ids:
                    logger.info(f' . Skip No.{index}. Not In Model ID List.')
                    continue

                if re.fullmatch(f'model_size_threshold_(\d+)', status['flag']):
                    origin_repo_size_threshold = int(re.fullmatch(f'model_size_threshold_(\d+)', status['flag']).group(1))
                    if model_size_threshold and model_size_threshold <= origin_repo_size_threshold:
                        model_ids = model_ids - {status['model_name']}
                        logger.info(f' . Skip No.{index}. This Model Converted Before, But Exceed The Threshold.')
                else:
                    logger.info(f' . Skip No.{index}. This Model Converted Before With Status: \"{status["flag"]}\".')
                    model_ids = model_ids - {status['model_name']}
    else:
        logger.info(f'-> Not Found Existing Status Files')

    logger.info(f'-> Instances Creating ...')
    for index, model_id in enumerate(sorted(model_ids), start=1):
        try:
            infered_model_size = infer_model_size(model_id)
        except Exception as error:
            logger.error(f' # No.{index} Model ID = {model_id}: Cannot Get The Model. Access Maybe Requested - {error}')
            save_status(status_filepath, dict(model_name=model_id, flag='access_error'))
            continue

        if model_size_threshold is None:
            pass
        else:
            if model_size_threshold < infered_model_size:
                logger.warning(f' # No.{index} Model ID = {model_id}: Model Size {get_human_readable_size_representation(infered_model_size)} Larger Than Threshold! Skip.')
                save_status(status_filepath, dict(model_name=model_id, flag=f'model_size_threshold_{model_size_threshold}'))
                continue

        logger.info(f'-> Infered Model Size = {get_human_readable_size_representation(infered_model_size)}')

        logger.info(f' # No.{index} Model ID = {model_id}: Now Converting ...') 
        logger.info(f'   v Converting HuggingFace Model into ONNX:')
        flag, onnx_model_filepaths = support_convert_method[framework](model_id, convert_cache_dirpath, huggingface_cache_dirpath, hf_file_system, device)
        if flag == 'success':
            pass
        else:
            logger.warning(f'   - Conversion Not Success - Flag: {flag}.')
            save_status(status_filepath, dict(model_name=model_id, flag=flag))
            clean_all_cache(model_id, convert_cache_dirpath, huggingface_cache_dirpath)
            continue
        logger.info(f'   ^ Finished.')

        model_info = None
        readme = None
        card_data = None
        annotations = None
        try:
            model_info = get_huggingface_model_info(model_id)
            readme = get_huggingface_model_readme(model_id, hf_file_system)
            card_data = get_huggingface_model_card_data_from_readme(readme)
            annotations = get_heuristic_annotations(model_id, card_data)
        except Exception as exception:
            logger.warning(f'   -> No Heuristic Annotations: {exception}')

        logger.info(f'   v Converting ONNX Model into NetworkX ...')
        for convert_index, onnx_model_filepath in enumerate(onnx_model_filepaths, start=1):
            try:
                instance = Instance(
                    model=onnx_model_filepath,
                    labels=dict(
                        model_source='HuggingFace',
                        model_name=model_id,
                        onnx_model_filename=onnx_model_filepath.name,
                        download=model_info['downloads'],
                        like=model_info['likes'],
                        tag=model_info['tags'],
                        readme=readme,
                        annotations=annotations
                    )
                )
                instance_save_dirpath = save_dirpath.joinpath(get_instance_dirname(model_id.replace('/', '--HF--'), 'HuggingFace', f'{onnx_model_filepath.stem}-{convert_index}'))
                instance.save(instance_save_dirpath)
                logger.info(f'     ┌ No.{convert_index} Converted')
                logger.info(f'     | From: {onnx_model_filepath}')
                logger.info(f'     └ Save: {instance_save_dirpath}')
                flag = 'success'
            except Exception as exception:
                logger.info(f'     ┌ No.{convert_index} Error')
                logger.error(f'    └ [ONNX -> NetworkX Error] OR [Instance Saving Error] - {exception}')
                flag = 'fail'
        logger.info(f'   ^ Converted.')
        save_status(status_filepath, dict(model_name=model_id, flag=flag))

        clean_all_cache(model_id, convert_cache_dirpath, huggingface_cache_dirpath)

    logger.info(f'-> Instances Created.')
