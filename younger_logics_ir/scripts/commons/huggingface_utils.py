#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-27 16:49:36
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import re
import tqdm
import pathlib
import requests

from typing import Any, Iterable, Generator
from huggingface_hub import utils, HfFileSystem, ModelCard, ModelCardData, get_hf_file_metadata, hf_hub_url, scan_cache_dir
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from yaml.scanner import ScannerError

from younger.commons.io import delete_dir
from younger.commons.logging import logger


HUGGINGFACE_HUB_API_ENDPOINT = 'https://huggingface.co/api'


def get_data_from_huggingface_hub_api(path: str, params: dict | None = None, token: str | None = None) -> Generator[Any, None, None]:
    """
    _summary_

    :param path: _description_
    :type path: str
    :param params: _description_, defaults to None
    :type params: dict | None, optional
    :param token: _description_, defaults to None
    :type token: str | None, optional

    :yield: _description_
    :rtype: Generator[Any, None, None]

    .. note::
        The Code are modified based on the official Hugging Face Hub source codes. (https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_pagination.py)
        `paginate` is called by list_models, list_datasets, and list_spaces methods of huggingface_hub.HfApi;
    """

    params = params or dict()

    session = requests.Session()
    headers = utils.build_hf_headers(token=token)
    response = session.get(path, params=params, headers=headers)
    utils.hf_raise_for_status(response)
    yield from response.json()

    # Follow pages
    # Next link already contains query params
    next_page_path = response.links.get("next", {}).get("url")
    while next_page_path is not None:
        logger.debug(f"Pagination detected. Requesting next page: {next_page_path}")
        response = session.get(next_page_path, headers=headers)
        utils.hf_raise_for_status(response)
        yield from response.json()
        next_page_path = response.links.get("next", {}).get("url")


def get_huggingface_model_ids(token: str | None = None) -> list[str]:
    models_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/models'
    models = get_data_from_huggingface_hub_api(models_path, params=dict(sort='lastModified', direction=-1, expand=['lastModified']), token=token)
    model_ids = [model['id'] for model in models]
    return model_ids


def get_huggingface_metric_ids(token: str | None = None) -> list[str]:
    metrics_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/metrics'
    metrics = get_data_from_huggingface_hub_api(metrics_path, token=token)
    metric_ids = [metric['id'] for metric in metrics]
    return metric_ids


def get_huggingface_task_ids(token: str | None = None) -> list[str]:
    tasks_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/tasks'
    tasks = get_data_from_huggingface_hub_api(tasks_path, token=token)
    task_ids = [task_id for task_id, task_info in tasks.items()]
    return task_ids


def get_huggingface_model_infos(token: str | None = None) -> list[dict[str, Any]]:
    models_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/models'

    model_infos: list[dict[str, Any]] = list()
    logger.info(f' v Retrieving All Model Infos ...')
    model_ids = get_huggingface_model_ids(token=token)
    with tqdm.tqdm(total=len(model_ids), desc='Retrieve Model') as progress_bar:
        for model_id in model_ids:
            progress_bar.set_description(f'Retrieve Model - {model_id}')
            model_info = get_data_from_huggingface_hub_api(f'{models_path}/{model_id}', params=dict(expand=['cardData', 'lastModified', 'likes', 'downloadsAllTime', 'siblings', 'tags']), token=token)
            model_infos.append(model_info)
            progress_bar.update(1)
    logger.info(f'   Retrieved.')
    logger.info(f' ^ Total = {len(model_infos)}.')

    return model_infos


def get_huggingface_metric_infos(token: str | None = None) -> list[dict[str, Any]]:
    metrics_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/metrics'
    metrics = get_data_from_huggingface_hub_api(metrics_path, token=token)
    metric_infos = [metric for metric in metrics]
    return metric_infos


def get_huggingface_task_infos(token: str | None = None) -> list[dict[str, Any]]:
    tasks_path = f'{HUGGINGFACE_HUB_API_ENDPOINT}/tasks'
    tasks = get_data_from_huggingface_hub_api(tasks_path, token=token)
    task_infos = [task_info for task_id, task_info in tasks.items()]
    return task_infos


def get_huggingface_model_readmes(model_ids: list[str], ignore_errors: bool = False) -> Generator[str, None, None]:
    hf_file_system = HfFileSystem()
    for model_id in model_ids:
        try:
            yield get_huggingface_model_readme(model_id, hf_file_system)
        except Exception as exception:
            if ignore_errors:
                continue
            else:
                raise exception


def get_huggingface_model_readme(model_id: str, hf_file_system: HfFileSystem) -> str:
    if hf_file_system.exists(f'{model_id}/README.md'):
        try:
            readme_file_size = hf_file_system.size(f'{model_id}/README.md')
            if 1*1024*1024*1024 < readme_file_size:
                logger.error(f"REPO: {model_id}. Strange README File Error - File Size To Large: {convert_bytes(readme_file_size)}")
                raise MemoryError
            with hf_file_system.open(f'{model_id}/README.md', mode='r', encoding='utf-8') as readme_file:
                readme = readme_file.read()
                readme = readme.replace('\t', ' ')
                return readme
        except UnicodeDecodeError as error:
            logger.error(f"REPO: {model_id}. Encoding Error - The Encoding [UTF-8] are Invalid. - Error: {error}")
            raise error
        except Exception as exception:
            logger.error(f"REPO: {model_id}. Encounter An Error {exception}.")
            raise exception
    else:
        logger.info(f"REPO: {model_id}. No README.md, skip.")
        raise FileNotFoundError


def get_huggingface_model_card_data(model_id: str, hf_file_system: HfFileSystem) -> ModelCardData:
    readme = get_huggingface_model_readme(model_id, hf_file_system)
    return get_huggingface_model_card_data_from_readme(readme)


def get_huggingface_model_card_data_from_readme(readme: str) -> ModelCardData:
    try:
        return ModelCard(readme, ignore_metadata_errors=True).data
    except ScannerError as error:
        logger.error(f' !!! Return Empty Card !!! Format of YAML at the Begin of README File Maybe Wrong. Error: {error}')
        raise error
    except ValueError as error:
        logger.error(f' !!! YAML ValueError !!! Format of YAML at the Begin of README File Maybe Wrong. Error: {error}')
        raise error
    except Exception as exception:
        logger.error(f' !!! Unknow ModelCard Parse Error !!! Format of YAML at the Begin of README File Maybe Wrong. Error: {exception}')
        raise exception


def infer_huggingface_hub_model_size(model_id: str) -> int:
    hf_file_system = HfFileSystem()
    filenames = list()
    filenames.extend(hf_file_system.glob(model_id+'/*.bin'))
    filenames.extend(hf_file_system.glob(model_id+'/*.pb'))
    filenames.extend(hf_file_system.glob(model_id+'/*.h5'))
    filenames.extend(hf_file_system.glob(model_id+'/*.hdf5'))
    filenames.extend(hf_file_system.glob(model_id+'/*.ckpt'))
    filenames.extend(hf_file_system.glob(model_id+'/*.keras'))
    filenames.extend(hf_file_system.glob(model_id+'/*.msgpack'))
    filenames.extend(hf_file_system.glob(model_id+'/*.safetensors'))
    infered_model_size = 0
    for filename in filenames:
        meta_data = get_hf_file_metadata(hf_hub_url(repo_id=model_id, filename=filename[len(model_id)+1:]))
        infered_model_size += meta_data.size
    return infered_model_size


def clean_huggingface_hub_cache_root(cache_dirpath: pathlib.Path):
    info = scan_cache_dir(cache_dirpath)
    commit_hashes = list()
    for repo in list(info.repos):
        for revision in list(repo.revisions):
            commit_hashes.append(revision.commit_hash)
    delete_strategy = info.delete_revisions(*commit_hashes)
    delete_strategy.execute()


def clean_model_default_cache(model_id: str):
    clean_model_specify_cache(model_id, pathlib.Path(HUGGINGFACE_HUB_CACHE))


def clean_model_specify_cache(model_id: str, specify_cache_dirpath: pathlib.Path):
    model_id = model_id.replace("/", "--")

    model_cache_dirpath = specify_cache_dirpath.joinpath(f"models--{model_id}")

    if model_cache_dirpath.is_dir():
        delete_dir(model_cache_dirpath)


def get_huggingface_model_file_indicators(model_id: str, suffixes: list[str] | None = None) -> list[tuple[str, str]]:
    hf_file_system = HfFileSystem()
    model_file_indicators = list()
    for dirpath, dirnames, filenames in hf_file_system.walk(model_id):
        for filename in filenames:
            if suffixes is None:
                model_file_indicators.append((dirpath[len(model_id)+1:], filename))
            else:
                if os.path.splitext(filename)[1] in suffixes:
                    model_file_indicators.append((dirpath[len(model_id)+1:], filename))
    return model_file_indicators