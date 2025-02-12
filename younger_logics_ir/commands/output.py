#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-12-09 09:27:45
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-01-08 10:03:58
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib

from younger_logics_ir.commands import equip_logger


@click.group(name='output')
def output():
    pass

@output.command(name='filter')
@click.option('--load-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be loaded.')
@click.option('--save-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--opset-version',    required=False, type=int, default=None, help='Used to specify the instances that are equal to the specified version.')
@click.option('--worker-number',    required=False, type=int, default=1, help='The number of workers to use for processing.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def output_filter(
    load_dirpath,
    save_dirpath,
    opset_version,
    worker_number,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.bootstrap import filter

    filter.main(load_dirpath, save_dirpath, opset_version=opset_version, worker_number=worker_number)


@output.command(name='clean')
@click.option('--load-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be loaded.')
@click.option('--save-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def output_clean(
    load_dirpath,
    save_dirpath,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.bootstrap import clean

    clean.main(load_dirpath, save_dirpath)


@output.command(name='statistics')
@click.option('--load-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be loaded.')
@click.option('--save-dirpath',     required=True,  type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path), help='The directory where the data will be saved.')
@click.option('--granularity',      required=True,  type=click.Choice(['Instance', 'LogicX'], case_sensitive=True), help='Indicates the type of data that needs to be statistically analyzed.')
@click.option('--plot',             is_flag=True,   help='Indicates whether the statistics should be visualized.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def output_statistics(
    load_dirpath,
    save_dirpath,
    granularity,
    plot,
    logging_filepath,
):
    equip_logger(logging_filepath)

    from younger_logics_ir.scripts.bootstrap import statistics

    statistics.main(load_dirpath, save_dirpath, granularity, plot)
