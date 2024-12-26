#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-10 11:10:18
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-13 14:28:53
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import json
import tqdm
import pathlib
import multiprocessing

from typing import Literal
from huggingface_hub import login, HfFileSystem

from younger.commons.io import load_json, save_json
from younger.commons.logging import logger

from younger_logics_ir.scripts.commons.huggingface_utils import get_huggingface_model_infos, get_huggingface_model_ids, get_huggingface_tasks, check_huggingface_model_eval_results


def get_huggingface_model_label_status(model_id: str) -> tuple[str, bool]:
    return (model_id, check_huggingface_model_eval_results(model_id, HfFileSystem()))


def save_huggingface_model_infos(save_dirpath: pathlib.Path, library: str | None = None, token: str | None = None, force_reload: bool | None = None, worker_number: int = 10):
    login(token=token)
    suffix = f'_{library}.json' if library else '.json'
    save_filepath = save_dirpath.joinpath(f'model_infos{suffix}')
    if save_filepath.is_file() and not force_reload:
        model_infos = load_json(save_filepath)
        logger.info(f' -> Already Retrieved. Total {len(model_infos)} Model Infos{f" (Library - {library})" if library else ""}. Results From: \'{save_filepath}\'.')
    else:
        filter_list = [library] if library else None
        model_infos = list(get_huggingface_model_infos(filter_list=filter_list, full=True, config=True, token=token))
        logger.info(f' -> Total {len(model_infos)} Model Infos{f" (Library - {library})" if library else ""}.')
        logger.info(f' v Saving Results Into {save_filepath} ...')
        save_json(model_infos, save_filepath, indent=2)
        logger.info(f' ^ Saved.')

    logger.info(f' -> Begin Retrieve Label')
    json_miwls_suffix = f'_{library}-with_label_status.json' if library else '-with_label_status.json'
    json_miwls_save_filepath = save_dirpath.joinpath(f'model_ids{json_miwls_suffix}')
    temp_miwls_suffix = f'_{library}-with_label_status.temp' if library else '-with_label_status.temp'
    temp_miwls_save_filepath = save_dirpath.joinpath(f'model_ids{temp_miwls_suffix}')
    if json_miwls_save_filepath.is_file() and not force_reload:
        model_ids_with_label_status = load_json(json_miwls_save_filepath)
        logger.info(f' -> Already Retrieved. Total {len(model_ids_with_label_status)} Model Ids With Label Status{f" (Library - {library})" if library else ""}. Results Saved In: \'{json_miwls_save_filepath}\'.')
    else:
        model_ids = set([model_info['id'] for model_info in model_infos])
        model_ids_with_label_status = dict()

        if temp_miwls_save_filepath.is_file():
            if force_reload:
                temp_miwls_save_filepath.unlink()
            else:
                with open(temp_miwls_save_filepath, 'r') as temp_miwls_save_file:
                    for line in temp_miwls_save_file:
                        model_id, label_status = json.loads(line)
                        model_ids_with_label_status[model_id] = label_status
                        model_ids.remove(model_id)
        else:
            temp_miwls_save_filepath.touch()

        logger.info(f' v Retrieving ...')
        with multiprocessing.Pool(worker_number) as pool:
            with tqdm.tqdm(total=len(model_ids)) as progress_bar:
                for index, (model_id, label_status) in enumerate(pool.imap_unordered(get_huggingface_model_label_status, model_ids), start=1):
                    model_ids_with_label_status[model_id] = label_status
                    with open(temp_miwls_save_filepath, 'a') as temp_miwls_save_file:
                        model_id_with_label_status = json.dumps([model_id, label_status])
                        temp_miwls_save_file.write(f'{model_id_with_label_status}\n')
                    progress_bar.update()
        logger.info(f' ^ Retrieved.')
        logger.info(f' v Saving Results Into {json_miwls_save_filepath} ...')
        save_json(model_ids_with_label_status, json_miwls_save_filepath, indent=2)
        logger.info(f' ^ Saved.')
    logger.info(f' => Finished')


def save_huggingface_model_ids(save_dirpath: pathlib.Path, library: str | None = None, token: str | None = None):
    model_ids = list(get_huggingface_model_ids(library, token=token))
    suffix = f'_{library}.json' if library else '.json'
    save_filepath = save_dirpath.joinpath(f'model_ids{suffix}')
    save_json(model_ids, save_filepath, indent=2)
    logger.info(f'Total {len(model_ids)} Model IDs{f" (Library - {library})" if library else ""}. Results Saved In: \'{save_filepath}\'.')


def save_huggingface_tasks(save_dirpath: pathlib.Path):
    tasks = get_huggingface_tasks()
    save_filepath = save_dirpath.joinpath('huggingface_tasks.json')
    save_json(tasks, save_filepath, indent=2)
    logger.info(f'Total {len(tasks)} Tasks. Results Saved In: \'{save_filepath}\'.')


def main(mode: Literal['Model_Infos', 'Model_IDs', 'Tasks'], save_dirpath: pathlib.Path, mirror_url: str, **kwargs):
    assert mode in {'Model_Infos', 'Model_IDs', 'Tasks'}

    os.environ['HF_ENDPOINT'] = 'https://huggingface.co/' if mirror_url == '' else mirror_url

    if mode == 'Model_Infos':
        save_huggingface_model_infos(save_dirpath, library=kwargs['library'], token=kwargs['token'], force_reload=kwargs['force_reload'], worker_number=kwargs['worker_number'])
        return

    if mode == 'Model_IDs':
        save_huggingface_model_ids(save_dirpath, library=kwargs['library'], token=kwargs['token'])
        return

    if mode == 'Tasks':
        save_huggingface_tasks(save_dirpath)
        return
