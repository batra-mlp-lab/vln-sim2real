# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Dynamic Time Warping based evaluation metrics for VLN.
   Includes several modifications from the original work:
   1. Modified to operate on predictions that are lists of cartesian (x,y) inputs,
   2. Reference paths are treated as line segments in distance
   3. DTW is normalized by the number of prediction points, not the number of reference points."""

from __future__ import print_function

import math
import numpy as np



class DTW(object):
  """Dynamic Time Warping (DTW) evaluation metrics. """

  def __init__(self, threshold=3.0, num_samples=100):
    """Initializes a DTW object.

    Args:
      threshold: distance threshold $d_{th}$ (float).
    """
    self.threshold = threshold
    self.num_samples = num_samples

  def distance(self, pos1, pos2):
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

  def resample_path(self, path):
    path_dists = np.cumsum(np.array([0]+[self.distance(a,b) for a,b in zip(path[:-1], path[1:])]))
    step_dist = path_dists[-1]/(self.num_samples-1)
    sampled_path = [path[0]]
    path_ix = 0
    for sample_ix in range(1, self.num_samples):
      if sample_ix + 1 == self.num_samples:
        sampled_path.append(path[-1])
      else:
        sample_dist = sample_ix * step_dist
        while sample_dist > path_dists[path_ix]:
          path_ix += 1
        prev = np.array(path[path_ix-1])
        next = np.array(path[path_ix])
        pt = prev + (sample_dist - path_dists[path_ix-1]) * (next - prev)
        sampled_path.append(pt)
    return sampled_path   

  def __call__(self, prediction, reference, metric='sdtw'):
    """Computes DTW metrics.

    Args:
      prediction: list of (x,y) (float) pairs, path predicted by agent.
      reference: list of (x,y) (float) pairs, the ground truth path.
      metric: one of ['ndtw', 'sdtw', 'dtw'].

    Returns:
      the DTW between the prediction and reference path (float).
    """
    assert metric in ['ndtw', 'sdtw', 'dtw']

    reference = self.resample_path(reference)
    prediction = self.resample_path(prediction)

    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
      for j in range(1, len(reference)+1):
        best_previous_cost = min(
            dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
        cost = self.distance(prediction[i-1], reference[j-1])
        dtw_matrix[i][j] = cost + best_previous_cost
    dtw = dtw_matrix[len(prediction)][len(reference)]

    if metric == 'dtw':
      return dtw

    ndtw = np.exp(-dtw/(self.threshold * len(reference)))
    if metric == 'ndtw':
      return ndtw

    success = self.distance(prediction[-1], reference[-1]) <= self.threshold
    return success * ndtw

        


