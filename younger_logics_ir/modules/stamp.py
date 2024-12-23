#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:44
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2024-12-09 09:29:33
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from younger.commons.hash import hash_string
from younger.commons.version import semantic_release, check_semantic, str_to_sem


class Stamp(object):
    def __init__(self, version: str, checksum: str) -> None:
        assert check_semantic(version), f'The version provided must follow the SemVer 2.0.0 Specification.'
        self._version = str_to_sem(version)

        assert isinstance(checksum, str), f'Invalid type of checksum value, should be \"str\", instead \"{type(checksum)}\"'
        self._checksum = checksum

    @property
    def version(self) -> semantic_release.Version:
        return self._version

    @property
    def checksum(self) -> str:
        return self._checksum

    @property
    def dict(self) -> dict:
        return dict(
            version = str(self.version),
            checksum = self.checksum,
        )

    def __repr__(self):
        return (
            f'Version: {str(self.version)}\n'
            f'Checksum: {self.checksum}\n'
        )

    def __hash__(self):
        return hash(hash_string(self.checksum))

    def __eq__(self, stamp: 'Stamp') -> bool:
        return self.checksum == stamp.checksum

    def __lt__(self, stamp: 'Stamp') -> bool:
        return self.version < stamp.version
