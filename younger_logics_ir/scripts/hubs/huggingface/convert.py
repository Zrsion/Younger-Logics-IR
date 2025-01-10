#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-10 10:16:06
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import onnx
import tqdm
import pathlib
import multiprocessing

from typing import Any, Literal, Callable

from huggingface_hub import login, hf_hub_download, snapshot_download, utils

from younger.commons.io import saves_json, loads_json, create_dir, delete_dir, get_human_readable_size_representation, load_json
from younger.commons.hash import hash_string
from younger.commons.logging import logger

from younger_logics_ir.modules import Instance, Implementation, Origin
from younger_logics_ir.converters import convert
from younger_logics_ir.converters.onnx2ir.io import load_model

from younger_logics_ir.commons.constants import YLIROriginHub

from younger_logics_ir.scripts.commons.utils import get_onnx_opset_versions, get_onnx_model_opset_version

from .utils import get_huggingface_hub_model_siblings, clean_huggingface_hub_model_cache


def clean_cache(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path):
    delete_dir(cvt_cache_dirpath, only_clean=True)
    clean_huggingface_hub_model_cache(model_id, ofc_cache_dirpath)


def safe_optimum_export(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, results_queue: multiprocessing.Queue, device: str):
    from optimum.exporters.onnx import main_export
    this_status: dict[int, Any] = dict()
    this_instances: list[Instance] = list()
    for onnx_opset_version in get_onnx_opset_versions():
        try:
            main_export(model_id, cvt_cache_dirpath, opset=onnx_opset_version, device=device, cache_dir=ofc_cache_dirpath, monolith=True, do_validation=False, trust_remote_code=True, no_post_process=True)
            this_status[onnx_opset_version] = dict()
            for filepath in cvt_cache_dirpath.iterdir():
                filename = filepath.name
                try:
                    if filepath.suffix == '.onnx':
                        instance = Instance()
                        instance.setup_logicx(convert(load_model(filename)))
                        this_instances.append(instance)
                    this_status[onnx_opset_version][filename] = 'success'
                except:
                    this_status[onnx_opset_version][filename] = 'onnx2logicx_convert_error'
        except MemoryError as exception:
            this_status[onnx_opset_version] = 'optimum2onnx_oversize'
        except utils.RepositoryNotFoundError as exception:
            this_status[onnx_opset_version] = 'optimum2onnx_access_deny'
        except Exception as exception:
            this_status[onnx_opset_version] = 'optimum2onnx_convert_error'
        delete_dir(cvt_cache_dirpath, only_clean=True)

    results_queue.put((this_status, this_instances))


def convert_optimum(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]:
    assert device in {'cpu', 'cuda'}
    status: dict[str, dict[int, str] | Literal['system_kill']] = dict()
    instances: list[Instance] = list()

    results_queue = multiprocessing.Queue()
    subprocess = multiprocessing.Process(target=safe_optimum_export, args=(model_id, cvt_cache_dirpath, ofc_cache_dirpath, results_queue, device))
    subprocess.start()
    subprocess.join()

    if results_queue.empty():
        this_status = 'system_kill'
        this_instances = list()
    else:
        # this_status: dict[int, str]
        # this_instances: list[Instance]
        this_status, this_instances = results_queue.get()

    status['main_model'] = this_status
    instances.extend(this_instances)

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances


def safe_keras_export(cvt_cache_dirpath: pathlib.Path, keras_model_path: pathlib.Path, results_queue: multiprocessing.Queue):
    from .miscs import tf2onnx_main_export
    this_status: dict[int, str] = dict()
    this_instances: list[Instance] = list()
    onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(keras_model_path)}.onnx')

    if keras_model_path.is_dir():
        model_type = 'saved_model'
    if keras_model_path.is_file():
        model_type = 'keras'

    for onnx_opset_version in get_onnx_opset_versions():
        try:
            tf2onnx_main_export(keras_model_path, onnx_model_path, onnx_opset_version, model_type=model_type)
            try:
                instance = Instance()
                instance.setup_logicx(convert(load_model(onnx_model_path)))
                this_instances.append(instance)
                this_status[onnx_opset_version] = 'success'
            except Exception as exception:
                this_status[onnx_opset_version] = 'onnx2logicx_convert_error'
        except Exception as exception:
            this_status[onnx_opset_version] = 'keras2onnx_convert_error'
        delete_dir(cvt_cache_dirpath, only_clean=True)

    results_queue.put((this_status, this_instances))


def convert_keras(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]:
    status: dict[str, dict[int, str] | Literal['system_kill']] = dict()
    instances: list[Instance] = list()
    remote_keras_model_paths = list()
    for remote_keras_model_path in get_huggingface_hub_model_siblings(model_id, suffixes=['.keras', '.hdf5', '.h5', '.pbtxt', '.pb']):
        if remote_keras_model_path.endswith('.pbtxt') or remote_keras_model_path.endswith('.pb'):
            remote_keras_model_paths.append(os.path.dirname(remote_keras_model_path))
        else:
            remote_keras_model_paths.append(remote_keras_model_path)
    remote_keras_model_paths = list(set(remote_keras_model_paths))

    for remote_keras_model_path in remote_keras_model_paths:
        remote_keras_model_name = os.path.splitext(remote_keras_model_path)[0]
        if os.path.isdir(remote_keras_model_path):
            keras_model_path = pathlib.Path(snapshot_download(model_id, allow_patterns=f'{remote_keras_model_path}/*', cache_dir=ofc_cache_dirpath)).joinpath(remote_keras_model_path)
        else:
            keras_model_path = pathlib.Path(hf_hub_download(model_id, remote_keras_model_path, cache_dir=ofc_cache_dirpath))

        results_queue = multiprocessing.Queue()
        subprocess = multiprocessing.Process(target=safe_keras_export, args=(cvt_cache_dirpath, keras_model_path, results_queue))
        subprocess.start()
        subprocess.join()
        if results_queue.empty():
            this_status = 'system_kill'
            this_instances = list()
        else:
            # this_status: dict[int, str]
            # this_instances: list[Instance]
            this_status, this_instances = results_queue.get()

        status[remote_keras_model_name] = this_status
        instances.extend(this_instances)

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances


def safe_tflite_export(cvt_cache_dirpath: pathlib.Path, tflite_model_path: pathlib.Path, results_queue: multiprocessing.Queue):
    from .miscs import tf2onnx_main_export
    this_status: dict[int, str] = dict()
    this_instances: list[Instance] = list()
    onnx_model_path = cvt_cache_dirpath.joinpath(f'{hash_string(tflite_model_path)}.onnx')
    for onnx_opset_version in get_onnx_opset_versions():
        try:
            tf2onnx_main_export(tflite_model_path, onnx_model_path, onnx_opset_version, model_type='tflite')
            try:
                instance = Instance()
                instance.setup_logicx(convert(load_model(onnx_model_path)))
                this_instances.append(instance)
                this_status[onnx_opset_version] = 'success'
            except Exception as exception:
                this_status[onnx_opset_version] = 'onnx2logicx_convert_error'
        except Exception as exception:
            this_status[onnx_opset_version] = 'tflite2onnx_convert_error'
        delete_dir(cvt_cache_dirpath, only_clean=True)

    results_queue.put((this_status, this_instances))
        

def convert_tflite(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]:
    status: dict[str, dict[int, str] | Literal['system_kill']] = dict()
    instances: list[Instance] = list()
    remote_tflite_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.tflite'])
    for remote_tflite_model_path in remote_tflite_model_paths:
        remote_tflite_model_name = os.path.splitext(remote_tflite_model_path)[0]
        tflite_model_path = pathlib.Path(hf_hub_download(model_id, remote_tflite_model_path, cache_dir=ofc_cache_dirpath))
        results_queue = multiprocessing.Queue()
        subprocess = multiprocessing.Process(target=safe_tflite_export, args=(cvt_cache_dirpath, tflite_model_path, results_queue))
        subprocess.start()
        subprocess.join()
        if results_queue.empty():
            this_status = 'system_kill'
            this_instances = list()
        else:
            # this_status: dict[int, str]
            # this_instances: list[Instance]
            this_status, this_instances = results_queue.get()

        status[remote_tflite_model_name] = this_status
        instances.extend(this_instances)

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances


def safe_onnx_export(cvt_cache_dirpath: pathlib.Path, onnx_model_path: pathlib.Path, results_queue: multiprocessing.Queue):
    this_status: dict[int, str] = dict()
    this_instances: list[Instance] = list()
    onnx_model = load_model(onnx_model_path)
    onnx_model_opset_version = get_onnx_model_opset_version(onnx_model)
    for onnx_opset_version in get_onnx_opset_versions():
        if onnx_opset_version == onnx_model_opset_version:
            continue
        try:
            # Convert the ONNX model to the target opset version. This step will not occupy disk space. Thus cvt_cache_dirpath is not used.
            other_version_onnx_model = onnx.version_converter.convert_version(onnx_model, onnx_opset_version)
            try:
                instance = Instance()
                instance.setup_logicx(convert(other_version_onnx_model))
                this_instances.append(instance)
                this_status[onnx_opset_version] = 'success'
            except Exception as exception:
                this_status[onnx_opset_version] = 'onnx2logicx_convert_error'
        except Exception as exception:
            this_status[onnx_opset_version] = 'onnx2onnx_convert_error'
        delete_dir(cvt_cache_dirpath, only_clean=True)

    results_queue.put((this_status, this_instances))


def convert_onnx(model_id: str, cvt_cache_dirpath: pathlib.Path, ofc_cache_dirpath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]:
    status: dict[str, dict[int, str] | Literal['system_kill']] = dict()
    instances: list[Instance] = list()
    remote_onnx_model_paths = get_huggingface_hub_model_siblings(model_id, suffixes=['.onnx'])
    for remote_onnx_model_path in remote_onnx_model_paths:
        remote_onnx_model_name = os.path.splitext(remote_onnx_model_path)[0]
        onnx_model_path = pathlib.Path(hf_hub_download(model_id, remote_onnx_model_path, cache_dir=ofc_cache_dirpath))
        results_queue = multiprocessing.Queue()
        subprocess = multiprocessing.Process(target=safe_onnx_export, args=(cvt_cache_dirpath, onnx_model_path, results_queue))
        subprocess.start()
        subprocess.join()
        if results_queue.empty():
            this_status = 'system_kill'
            this_instances = list()
        else:
            # this_status: dict[int, str]
            # this_instances: list[Instance]
            this_status, this_instances = results_queue.get()

        status[remote_onnx_model_name] = this_status
        instances.extend(this_instances)

    clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)
    return status, instances


def get_model_infos_and_convert_method(model_infos_filepath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras'], model_size_limit: tuple[int, int]) -> tuple[list[dict[str, Any]], Callable[[str, pathlib.Path, pathlib.Path, Literal['cpu', 'cuda']], tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]]]:
    # This is the convert order.
    supported_frameworks: list[str] = ['optimum', 'keras', 'onnx']
    supported_convert_methods: dict[str, Callable[[str, pathlib.Path, pathlib.Path, Literal['cpu', 'cuda']], tuple[dict[str, dict[int, Any] | Literal['system_kill']], list[Instance]]]] = dict(
        optimum=convert_optimum,
        onnx=convert_onnx,
        keras=convert_keras,
        # tflite=convert_tflite,
    )

    def get_model_frameworks(model_tags: list[str]) -> Literal['optimum', 'onnx', 'keras']:
        candidate_frameworks = set(model_tags) & set(supported_frameworks)
        model_frameworks = list()
        for supported_framework in supported_frameworks:
            if supported_framework in candidate_frameworks:
                model_frameworks.append(supported_framework)
        return model_frameworks

    model_infos: list[dict[str, Any]] = list()
    for model_info in load_json(model_infos_filepath):
        model_frameworks = get_model_frameworks(model_info['tags'])
        if framework in model_frameworks and model_size_limit[0] <= model_info['usedStorage'] and model_info['usedStorage'] <= model_size_limit[1]:
            model_infos.append(model_info)

    convert_method = supported_convert_methods[framework]
    return model_infos, convert_method


def get_convert_status_and_last_handled_model_id(sts_cache_dirpath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras'], model_size_limit: tuple[int, int]) -> tuple[list[dict[str, dict[int, Any]]], str | None]:
    convert_status: dict[str, dict[int, Any]] = list()
    specific_status_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}.sts')
    if specific_status_filepath.is_file():
        with open(specific_status_filepath, 'r') as specific_status_file:
            convert_status: dict[str, dict[int, Any]] = [loads_json(line.strip()) for line in specific_status_file]

    last_handled_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}_last_handled.sts')
    if last_handled_filepath.is_file():
        with open(last_handled_filepath, 'r') as last_handled_file:
            model_id = last_handled_file.read().strip()
        last_handled_model_id = model_id
    else:
        last_handled_model_id = None
    return convert_status, last_handled_model_id


def set_convert_status_last_handled_model_id(sts_cache_dirpath: pathlib.Path, framework: Literal['optimum', 'onnx', 'keras'], model_size_limit: tuple[int, int], convert_status: dict[str, dict[str, Any]], model_id: str):
    convert_status_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}.sts')
    with open(convert_status_filepath, 'a') as convert_status_file:
        convert_status_file.write(f'{saves_json((model_id, convert_status))}\n')

    last_handled_filepath = sts_cache_dirpath.joinpath(f'{framework}_{model_size_limit[0]}_{model_size_limit[1]}_last_handled.sts')
    with open(last_handled_filepath, 'w') as last_handled_file:
        last_handled_file.write(f'{model_id}\n')


def main(
    model_infos_filepath: pathlib.Path,
    save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path,
    device: Literal['cpu', 'cuda'] = 'cpu',
    framework: Literal['optimum', 'onnx', 'keras'] = 'optimum',
    model_size_limit_l: int | None = None,
    model_size_limit_r: int | None = None,
    token: str | None = None,
):
    """
    Retrieve Metadata of HuggingFace Models and Save Them Into Files.

    :param model_ids_filepath: _description_
    :type model_ids_filepath: pathlib.Path
    :param save_dirpath: _description_
    :type save_dirpath: pathlib.Path
    :param cache_dirpath: _description_
    :type cache_dirpath: pathlib.Path
    :param device: _description_, defaults to 'cpu'
    :type device: Literal[&#39;cpu&#39;, &#39;cuda&#39;], optional
    :param framework: _description_, defaults to 'optimum'
    :type framework: Literal[&#39;optimum&#39;, &#39;onnx&#39;, &#39;keras&#39;], optional
    :param model_size_limit_l: _description_, defaults to None
    :type model_size_limit_l: int | None, optional
    :param model_size_limit_r: _description_, defaults to None
    :type model_size_limit_r: int | None, optional
    :param token: _description_, defaults to None
    :type token: str | None, optional

    In this project we have a concept called Origin. Origin is a tuple of (hub, owner, name).

    HuggingFace Hub is a place where people can share their models, datasets, and scripts.
    This project hardcodes the hub as 'HuggingFace'.
    The Naming Convention of the Mdoel ID on HuggingFace Hub follows the format: {owner}/{name}.
    Thus the Origin of a Implementation, often called as Model which is a instance of a LogicX a.k.a. Neural Network Architecture (NNA), on HuggingFace Hub is Origin('HuggingFace', owner, name).

    Model Infos are retrieved from the HuggingFace Hub and sorted with lastModified time in descending order, and saved into a JSON file by using command `younger-logics-ir create onnx retrieve huggingface --mode Model_Infos --save-dirpath ${SAVE_DIRPATH}`.

    .. note::
        The Instances are saved into the directory named as 'Instances-HuggingFace-{Framework}' under the save_dirpath.

    .. note::
        This project will never support the conversion from TFLite to LogicX, because TensorFlow-ONNX converter only support python version from 3.7 to 3.10, and our project requires python version at least 3.11.
        See: https://github.com/onnx/tensorflow-onnx
    """

    model_size_limit_l = model_size_limit_l or 0
    logger.info(f'   Model Size Left Limit: {get_human_readable_size_representation(model_size_limit_l)}.')

    model_size_limit_r = model_size_limit_r or 1024 * 1024 * 1024 * 1024
    logger.info(f'   Model Size Right Limit: {get_human_readable_size_representation(model_size_limit_r)}.')

    model_size_limit = (model_size_limit_l, model_size_limit_r)

    model_infos, convert_method = get_model_infos_and_convert_method(model_infos_filepath, framework, model_size_limit)

    # Instances
    instances_dirpath = save_dirpath.joinpath(f'Instances')
    create_dir(instances_dirpath)

    # Official
    ofc_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFOfc')
    create_dir(ofc_cache_dirpath)

    # Convert
    cvt_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFCvt')
    create_dir(cvt_cache_dirpath)

    # Status
    sts_cache_dirpath = cache_dirpath.joinpath(f'Cache-HFSts')
    create_dir(sts_cache_dirpath)
    convert_status, last_handled_model_id = get_convert_status_and_last_handled_model_id(sts_cache_dirpath, framework, model_size_limit)
    number_of_converted_models = len(convert_status)
    logger.info(f'-> Previous Converted Models: {number_of_converted_models}')

    if token is not None:
        logger.info(f'-> HuggingFace Token Provided. Now Logging In ...')
        login(token)
    else:
        logger.info(f'-> HuggingFace Token Not Provided. Now Accessing Without Token ...')

    logger.info(f'-> Instances Creating ...')
    with tqdm.tqdm(total=len(model_infos), desc='Create Instances') as progress_bar:
        for convert_index, model_info in enumerate(model_infos, start=1):
            model_id = model_info['id']
            if last_handled_model_id is not None:
                if model_id == last_handled_model_id:
                    last_handled_model_id = None
                progress_bar.set_description(f'Converted, Skip - {model_id}')
                progress_bar.update(1)
                continue

            status, instances = convert_method(model_id, cvt_cache_dirpath, ofc_cache_dirpath, device)
            set_convert_status_last_handled_model_id(sts_cache_dirpath, framework, model_size_limit, status, model_id)
            clean_cache(model_id, cvt_cache_dirpath, ofc_cache_dirpath)

            model_owner, model_name = model_id.split('/')
            for instance_index, instance in enumerate(instances, start=1):
                instance.insert_label(
                    Implementation(
                        origin=Origin(YLIROriginHub.HUGGINGFACE, model_owner, model_name),
                        like=model_info['likes'],
                        download=model_info['downloadsAllTime'],
                    )
                )
                instance.save(instances_dirpath.joinpath(instance.unique))

            progress_bar.set_description(f'Convert - {model_id}')
            progress_bar.update(1)

    logger.info(f'-> Instances Created.')
