#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 14:47:13
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-03-14 16:37:37
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import shutil
import pathlib

from younger.commons.io import create_dir, load_toml
from younger.commons.logging import logger


@click.command()
@click.option('--options-filepath', required=True, type=pathlib.Path)
def main(options_filepath: pathlib.Path):
    options = load_toml(options_filepath)
    # print(options)

    i_dirpaths = [pathlib.Path(i_dirpath) for i_dirpath in options['i_dirpaths']]

    o_dirpath: pathlib.Path = pathlib.Path(options['o_dirpath'])
    create_dir(o_dirpath)

    split_count = 0
    inner_count = 0
    total_ins_count = 0
    dupli_ins_count = 0

    n2p: dict[str, pathlib.Path] = dict()
    for i_dirpath in i_dirpaths:
        for ins_i_dirpath in i_dirpath.iterdir():
            total_ins_count += 1
            if ins_i_dirpath.name in n2p:
                dupli_ins_count += 1
            else:
                n2p[ins_i_dirpath.name] = ins_i_dirpath

    ins_i_dirpaths = [ins_i_dirpath for _, ins_i_dirpath in n2p.items()]

    logger.info(f'Total {total_ins_count} Instances.')
    logger.info(f'Total {dupli_ins_count} Instances Have Duplicate Name.')
    logger.info(f'Total {total_ins_count - dupli_ins_count} Instances Will Be Moved.')

    logger.info(f'Begin Move ...')
    split_dirpath = o_dirpath.joinpath(f'{options["o_pattern"]}_{split_count}')
    for ins_i_dirpath in ins_i_dirpaths:
        create_dir(split_dirpath)
        ins_o_dirpath = split_dirpath.joinpath(ins_i_dirpath.name)

        shutil.move(ins_i_dirpath, ins_o_dirpath)
        inner_count += 1

        if inner_count == options['split_size']:
            logger.info(f'#{split_count} Split of Instances Moved: {split_dirpath}')

            split_count += 1
            split_dirpath = o_dirpath.joinpath(f'{options["o_pattern"]}_{split_count}')
            inner_count = 0

    if inner_count != 0:
        logger.info(f'#{split_count} Split of Instances Moved: {split_dirpath}')
    logger.info(f'Total {split_count} Splits of Instances.')


if __name__ == '__main__':
    main()
