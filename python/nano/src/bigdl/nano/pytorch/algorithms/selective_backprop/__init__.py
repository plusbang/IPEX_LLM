#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This file is adapted from https://github.com/mosaicml/composer
# https://github.com/mosaicml/composer/
# blob/dev/composer/algorithms/selective_backprop/selective_backprop.py

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
"""`Selective Backprop <https://arxiv.org/abs/1910.00762>`_ prunes minibatches according
to the difficulty of the individual training examples, and only computes weight gradients
over the pruned subset, reducing iteration time and speeding up training.
"""

from bigdl.nano.pytorch.algorithms.selective_backprop.selective_backprop import SelectiveBackprop

__all__ = ['SelectiveBackprop']
