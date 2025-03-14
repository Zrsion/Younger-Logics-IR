#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 14:47:13
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-03-14 15:54:51
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib


from younger.commons.io import create_dir
from younger.commons.logging import logger


@click.command()
@click.option('--i-dirpath', required=True,  type=pathlib.Path)
@click.option('--o-dirpath', required=True,  type=pathlib.Path)
@click.option('--o-pattern', required=True,  type=str)
@click.option('--file-size', required=False, type=int, defualt=10000)
def main(i_dirpath: pathlib.Path, o_dirpath: pathlib.Path, o_pattern: str, file_size: int = 10000):
    i_filepaths = [i_filepath for i_filepath in i_dirpath.iterdir()]
    create_dir(o_dirpath)

    all_status = list()
    file_count = 0
    line_count = 0
    status_num = 0

    for i_filepath in i_filepaths:
        with open(i_filepath, 'r') as i_file:
            for line in i_file:
                status_num += 1
                all_status.append(line.strip()+'\n')
                line_count += 1

                if line_count == file_size:
                    with open(o_dirpath.joinpath(f'{o_pattern}_{file_count}.sts'), 'w') as o_file:
                        o_file.writelines(all_status)
                    logger.info(f'#{file_count} Status File Saved: {o_pattern}_{file_count}.sts')
                    file_count += 1

                    all_status = list()
                    line_count = 0

    if all_status:
        with open(o_dirpath.joinpath(f'{o_pattern}_{file_count}.sts'), 'w') as o_file:
            o_file.writelines(all_status)
        logger.info(f'#{file_count} Status File Saved: {o_pattern}_{file_count}.sts')
        file_count += 1

    logger.info(f'Total {status_num} Status Records.')
    logger.info(f'Total {file_count} Status Files.')

if __name__ == '__main__':
    main()
