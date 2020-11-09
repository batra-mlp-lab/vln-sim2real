''' Evaluation of agent trajectories against the simulator (nav graph) 
    ground truth '''

import os
import sys
import json
import math
from collections import defaultdict
import networkx as nx
import numpy as np

from dtw import DTW


BASE_DIR = 'src/vln_evaluation/'
GT = BASE_DIR + 'data/R2R_coda.json'
CONN = BASE_DIR + 'data/connectivity/'
WMAP_JSON_OUTPUT = BASE_DIR + 'data/bags/wmap/submit_coda_robot_wmap.json'
NOMAP_JSON_OUTPUT = BASE_DIR + 'data/bags/nomap/submit_coda_robot_nomap.json'


def load_nav_graph(directory):
    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    (_, _, filenames) = os.walk(directory).next()
    for filename in [f for f in filenames if f.endswith('_connectivity.json')]:
        with open(directory + filename) as f:
            scan = filename.split('_connectivity.json')[0]
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


class SimEvaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, gt_data, nav_graph_dir):
        self.error_margin = 3.0
        self.dtw = DTW(threshold=self.error_margin)
        self.graphs = load_nav_graph(nav_graph_dir)
        self.gt = {}
        self.instr_ids = []
        for item in gt_data:
            if 'scan' not in item:
                item['scan'] = 'yZVvKaJZghh'
            if 'trajectory' in item:
                item['trajectory'] = self.path_to_points(item['trajectory'], item['scan'])
            if 'path' in item:
                item['trajectory'] = self.path_to_points(item['path'], item['scan'])
            if 'instr_id' in item:
              self.gt[item['instr_id']] = item
            else:
              for i in range(3):
                self.gt['%s_%d' % (item['path_id'], i)] = item
            self.instr_ids += ['%d_%d' % (item['path_id'],i) for i in range(3)]
        self.instr_ids = set(self.instr_ids)
        self.distances = {}
        # compute all shortest paths
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))


    def nav_error(self, instr_id, path):
        ''' Shortcut for getting some numbers on the fly. '''
        gt = self.gt[int(instr_id.split('_')[0])]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        return self.distances[final_position][goal]


    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def path_to_points(self, path, scan):
        if isinstance(path[0], str) or isinstance(path[0], unicode):
          return [self.graphs[scan].node[ix]['position'][:2] for ix in path]
        elif len(path[0]) == 3:
          return [self.graphs[scan].node[ix]['position'][:2] for ix,h,e in path]
        elif len(path[0]) == 2: # trajectory consists of (x,y) points
          return path
        else:
          raise TypeError('unknown path format')

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). '''
        gt = self.gt[instr_id]
        path = self.path_to_points(path, gt['scan'])
        start_pos = gt['trajectory'][0]
        assert self.distance(start_pos, path[0]) < 0.1, 'Result trajectories should include the start position'
        goal_pos = gt['trajectory'][-1]
        final_position = path[-1]
        #APPROXIMATION - We just use a straight line distance here
        final_dist = self.distance(goal_pos, final_position)
        self.scores['nav_errors'].append(final_dist)
        min_dist = final_dist
        path_len = 0
        for i,pos in enumerate(path):
            dist = self.distance(goal_pos, pos)
            if dist < min_dist:
                min_dist = dist
            if i > 0:
                path_len += self.distance(pos, path[i-1])
        self.scores['oracle_errors'].append(min_dist)
        self.scores['trajectory_lengths'].append(path_len)
        if 'path' in gt:
          self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][gt['path'][0]][gt['path'][-1]])
        else:  # Assume the reference path is the shortest path to goal
          ref_len = np.array([self.distance(a,b) for a,b in zip(path[:-1],path[1:])]).sum()
          self.scores['shortest_path_lengths'].append(ref_len)
        # Add ntdw and sdtw
        self.scores['ndtw'].append(self.dtw(path, gt['trajectory'], metric='ndtw'))
        self.scores['sdtw'].append(self.dtw(path, gt['trajectory'], metric='sdtw'))

    def score(self, data):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        for item in data:
            item['instr_id'] = str(item['instr_id'])
            # Check against expected ids
            if item['instr_id'] in ['0_0', '0_1', '0_2', '9_0', '9_1', '9_2', '32_0', '32_1', '32_2']:
                continue
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                self._score_item(item['instr_id'], item['trajectory'])
        if len(instr_ids) != 0:
            print('Trajectories not provided for %d instruction ids: %s' % (len(instr_ids),instr_ids))
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])

        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])

        spls = []
        for err,length,sp in zip(self.scores['nav_errors'],self.scores['trajectory_lengths'],self.scores['shortest_path_lengths']):
            if err < self.error_margin:
                spls.append(sp/max(length,sp))
            else:
                spls.append(0)

        score_summary ={
            'length': np.average(self.scores['trajectory_lengths']),
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle success_rate': float(oracle_successes)/float(len(self.scores['oracle_errors'])),
            'success_rate': float(num_successes)/float(len(self.scores['nav_errors'])),
            'spl': np.average(spls),
            'sdtw': np.average(self.scores['sdtw']),
            'ndtw': np.average(self.scores['ndtw'])
        }

        assert score_summary['spl'] <= score_summary['success_rate']
        return score_summary, self.scores


def evaluate(result_path):
    print('Evaluating %s' % result_path)
    with open(GT) as f:
        gt_data = json.load(f)
    evaluator = SimEvaluation(gt_data, CONN)
    with open(result_path) as f:
        results = json.load(f)
    summary,scores = evaluator.score(results)
    print(summary)


def score_ntdw():
  ''' Calculate ntdw between different experimental settings for paper Table 3. '''
  EXPS = ['sim_results/baseline/submit_coda.json',
    'sim_results/baseline/submit_coda_theta_1.json',
    'sim_results/baseline/submit_coda_theta_2.json',
    'sim_results/baseline/submit_coda_theta_3.json',
    'sim_results/baseline_color_jittered/submit_coda_theta_1.json',
    'sim_results/baseline_color_jittered/submit_coda_theta_2.json',
    'sim_results/baseline_color_jittered/submit_coda_theta_3.json',
    'bags/wmap/submit_coda_robot_wmap.json',          # Robot with map
    'bags/nomap/submit_coda_robot_nomap.json']         # Robot no map

  ntdw = np.zeros((len(EXPS), len(EXPS)))
  scorer = DTW()
  print('\nCalculating ntdw matrix for %s' % EXPS)
  for i, ref_path in enumerate(EXPS):
    with open('src/vln_evaluation/data/%s' % ref_path) as f:
      ref = json.load(f)
    for item in ref:
      item['path_id'] = int(item['instr_id'].split('_')[0])
    evaluator = SimEvaluation(ref, CONN)
    for j, pred_path in enumerate(EXPS):
      with open('src/vln_evaluation/data/%s' % pred_path) as f:
        pred = json.load(f)
      summary,scores = evaluator.score(pred)
      ntdw[i,j] = summary['ndtw']
  print(ntdw)

if __name__ == '__main__':

    evaluate(WMAP_JSON_OUTPUT)
    evaluate(NOMAP_JSON_OUTPUT)
    score_ntdw()


