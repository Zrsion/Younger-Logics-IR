#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-19 22:08:34
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-13 16:40:27
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import re
import math

from younger_logics_ir.commons.constants import READMEPattern



def extract_table_related_metrics_from_readme(readme: str) -> list[dict[str, list[str]]]:

    def extract_cells(row: str) -> list[str]:
        cell_str = row.strip()
        cell_str = cell_str[ 1:  ] if len(cell_str) and cell_str[ 0] == '|' else cell_str
        cell_str = cell_str[  :-1] if len(cell_str) and cell_str[-1] == '|' else cell_str
        cells = [cell.strip() for cell in cell_str.split('|')]
        return cells

    readme = readme.strip() + '\n'
    table_related = list()
    for match_result in re.finditer(READMEPattern.TABLE, readme, re.MULTILINE):
        headers = match_result.group(1)
        headers = extract_cells(headers)
        rows = list()
        for row in match_result.group(3).strip().split('\n'):
            rows.append(extract_cells(row))
        table_related.append(
            dict(
                headers=headers,
                rows=rows
            )
        )

    return table_related


def extract_digit_related_metrics_from_readme(readme: str) -> list[str]:

    def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        intervals = sorted(intervals)
        new_intervals = list()
        start, end = (-1, -1)
        for interval in intervals:
            if end < interval[0]:
                new_interval = (start, end)
                new_intervals.append(new_interval)
                start, end = interval

            else:
                if end < interval[1]:
                    end = interval[1]

        new_intervals.append((start, end))
        return new_intervals[1:]

    intervals = list()
    for match_result in re.finditer(READMEPattern.DIGIT, readme, re.MULTILINE):
        start = match_result.start() - 32
        end = match_result.end() + 32
        intervals.append((start, end))

    intervals = merge_intervals(intervals)
    digit_related = list()
    for start, end in intervals:
        digit_context = ' '.join(readme[start:end].split())
        digit_related.append(digit_context)

    return digit_related


def filter_readme_filepaths(filepaths: list[str]) -> list[str]:
    readme_filepaths = list()
    pattern = re.compile(r'.*readme(?:\.[^/\\]*)?$', re.IGNORECASE)
    for filepath in filepaths:
        if re.match(pattern, filepath) is not None:
            readme_filepaths.append(filepath)

    return readme_filepaths