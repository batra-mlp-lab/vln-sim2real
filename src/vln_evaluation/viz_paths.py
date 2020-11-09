#!/usr/bin/env python
''' Visualize paths from the collected rosbags, and create a
    json file that can be scored, and run the evaluation. '''

from __future__ import print_function
import json
import math
import os
import rosbag
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
import matplotlib.patches as patches


from eval import load_nav_graph,SimEvaluation
from dtw import DTW

BASE_DIR = 'src/vln_evaluation/'
WMAP_BAG_DIR = BASE_DIR + 'data/bags/wmap/'
NOMAP_BAG_DIR = BASE_DIR + 'data/bags/nomap/'
MAP = 'src/vln_agent/maps/coda-viz.jpg'
CONN = BASE_DIR + 'data/connectivity/'
GT = BASE_DIR + 'data/R2R_coda.json'
SIM = BASE_DIR + 'data/sim_results/baseline/submit_coda.json'
SIM_THETA = BASE_DIR + 'data/sim_results/baseline_color_jittered/submit_coda_theta_1.json'
OUT_DIR = BASE_DIR + 'figs/'
WMAP_JSON_OUTPUT = WMAP_BAG_DIR + 'submit_coda_robot_wmap.json'
NOMAP_JSON_OUTPUT = NOMAP_BAG_DIR + 'submit_coda_robot_nomap.json'


# World to image transforms for CODA
pixel_origin = (865.0,541.0)
pixel_scale = (907.0-422.0)/13.7787 # pixels to a meter
im_width = 1160.0
im_height = 1050.0

PATH_OFFSET=0.25
MARKER_SIZE=200


def load_instructions(filename):
    result = {}
    with open(filename) as f:
        data = json.load(f)
        for item in data:
            # Split multiple instructions into separate entries
            for j,instr in enumerate(item['instructions']):
                new_id = '%s_%d' % (item['path_id'], j)
                new_item = dict(item)
                new_item['instr_id'] = new_id
                new_item['instructions'] = instr
                new_item['path'] = item['path']
                new_item['heading'] = item['heading']
                result[new_id] = new_item
    return result,data


def load_sim_results(filename):
    result = {}
    with open(filename) as f:
        data = json.load(f)
        for item in data:
            result[item['instr_id']] = item
    return result,data


def quaternion_to_euler(q):

    x=q.x; y=q.y; z=q.z; w=q.w

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z



def read_bag(filename):
    bag = rosbag.Bag(filename)
    print (bag)

    results = []
    all_results = []
    for topic, msg, t in bag.read_messages(topics=['/agent/result', '/agent_relay/result']):
        if msg.reason == 'predicted stop action':
            results.append(msg)
        else:
            print('Skipping result with reason: %s' % msg.reason)
        all_results.append(msg)
    print('Found %d completed episodes' % len(results))

    poses = []
    for i,result in enumerate(results):
        poses.append([])
        for topic, msg, t in bag.read_messages(topics=['/amcl_pose'], start_time=result.start_time, end_time=result.end_time):
            poses[-1].append(msg)
        print('Found %d poses for episode %d' % (len(poses[-1]),i))
    bag.close()
    return results,poses,all_results


def img_extent():
    ''' The extent of the image in world coordinates '''
    left = -pixel_origin[0]/pixel_scale
    right = (im_width-pixel_origin[0])/pixel_scale
    top = pixel_origin[1]/pixel_scale
    bottom = -(im_height-pixel_origin[1])/pixel_scale
    return [left, right, bottom, top]


def poses_to_line(poses, offset=(0,0)):
    x = []
    y = []
    for pose in poses:
        x.append(pose.pose.pose.position.x+offset[0])
        y.append(pose.pose.pose.position.y+offset[1])
    return Line2D(x,y), (x[0],y[0]), (x[-1],y[-1])


def points_to_line(points, offset=(0,0)):
    x = []
    y = []
    for x_point, y_point in points:
        x.append(x_point+offset[0])
        y.append(y_point+offset[1])
    return Line2D(x,y), (x[0],y[0]), (x[-1],y[-1])


def path_to_line(path, graph, offset=(0,0)):
    x = []
    y = []
    for viewpoint in path:
        pos = graph.node[viewpoint]['position']
        x.append(pos[0]+offset[0])
        y.append(pos[1]+offset[1])
    return Line2D(x,y), (x[0],y[0]), (x[-1],y[-1])


def world_to_image(x,y):
    ''' Convert world coordinates into pixel coordinates '''
    return(pixel_origin[0]+pixel_scale*x, pixel_origin[1]+pixel_scale*y)


def get_arrow(angle):
    ''' Get an arrow representing this heading (in sim coords) '''
    a = angle
    ar = np.array([[-.25,-.5],[.25,-.5],[0,.5],[-.25,-.5]]).T
    rot = np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    return np.dot(rot,ar).T


def plot_all_trajectories(bagfiles, filename='AllPaths', json_output_file=None):

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    graph = load_nav_graph(CONN)['yZVvKaJZghh']
    gt,_ = load_instructions(GT)
    sim,_ = load_sim_results(SIM)
    sim_theta,_ = load_sim_results(SIM_THETA)
    img = plt.imread(MAP)

    result_output = {}
    result_pubs = []

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.imshow(img, extent=img_extent())
    ax.set_xlim([-25,10])
    ax.set_ylim([-15,15])
    ax.set_xlabel('m')
    ax.set_ylabel('m', rotation='horizontal')

    for bagfile in bagfiles:

        results,poses,ar = read_bag(bagfile)
        result_pubs += ar
        for result,pose in zip(results,poses):

            # Robot path
            robot_line, robot_start, robot_end = poses_to_line(pose, offset=(0,-PATH_OFFSET))
            r,p,y = quaternion_to_euler(pose[-1].pose.pose.orientation)
            end_heading = 0.5*math.pi - y
            r,p,y = quaternion_to_euler(pose[0].pose.pose.orientation)
            start_heading = 0.5*math.pi - y
            robot_line.set_color('red')
            robot_line.set_label('Robot trajectories')
            ax.add_line(robot_line)
            ax.scatter([robot_end[0]], [robot_end[1]], marker=get_arrow(end_heading), zorder=3, color='red', s=MARKER_SIZE, edgecolors='k', label='Start/end heading')
            r,p,y = quaternion_to_euler(pose[0].pose.pose.orientation)
            robot_start_heading = 0.5*math.pi - y
            ax.scatter([robot_start[0]], [robot_start[1]], marker=get_arrow(robot_start_heading), zorder=3, color='red', s=MARKER_SIZE, edgecolors='k')

            # Collect these into a results struct
            instr_id = result.instr_id
            start_id = gt[instr_id]['path'][0]
            start_pos = graph.node[start_id]['position'][:2]
            result_output[instr_id] = {
              'instr_id': instr_id,
              'trajectory': [tuple(start_pos)],
              'start_heading': start_heading,
              'end_heading': end_heading, 
            }
            for p in pose:
              result_output[instr_id]['trajectory'].append((p.pose.pose.position.x, p.pose.pose.position.y))

    robot_marker = Line2D([], [], color='red', marker=get_arrow(0), label='Robot trajectory', markersize=10)
    ax.legend(handles=[robot_marker], loc='lower left', fontsize='small')
    #ax.legend(loc='lower left', fontsize='small')
    plt.savefig('%s%s.png' % (OUT_DIR, filename))
    plt.waitforbuttonpress(0) 
    plt.close()

    result_pubs.sort(key=lambda x: x.header.stamp.to_sec())
    print('Summary of %d' % len(result_pubs))
    for result in result_pubs:
        print('%s: %s, %.f' % (result.instr_id,result.reason,result.header.stamp.to_sec()))

    count = 0
    print('instruction_params.yaml')
    for result in result_pubs:
        if result.reason == 'predicted stop action':
            print('- \'%s\'' % result.instr_id)
            count += 1
    print('Total of %d completed episodes' % count)

    result_output = result_output.values()
    if json_output_file:
        with open(json_output_file, 'w') as f:
            json.dump(result_output,f)
        print('Saved %d results to %s' % (len(result_output), json_output_file))


def plot_individual_trajectories():

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    graph = load_nav_graph(CONN)['yZVvKaJZghh']
    gt,gt_data = load_instructions(GT)
    sim,sim_data = load_sim_results(SIM)
    sim_theta,sim_theta_data = load_sim_results(SIM_THETA)
    robot_wmap,robot_wmap_data = load_sim_results(WMAP_JSON_OUTPUT)
    robot_nomap,robot_nomap_data = load_sim_results(NOMAP_JSON_OUTPUT)
    img = plt.imread(MAP)

    evaluator = SimEvaluation(gt_data, CONN)
    sim_data.sort(key=lambda x: x['instr_id'])
    _, sim_scores = evaluator.score(sim_data)
    sim_theta_data.sort(key=lambda x: x['instr_id'])
    _, sim_theta_scores = evaluator.score(sim_theta_data)
    robot_wmap_data.sort(key=lambda x: x['instr_id'])
    _, robot_wmap_scores = evaluator.score(robot_wmap_data)
    robot_nomap_data.sort(key=lambda x: x['instr_id'])    
    _, robot_nomap_scores = evaluator.score(robot_nomap_data)

    for i,instr_id in enumerate([item['instr_id'] for item in robot_wmap_data]):

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.imshow(img, extent=img_extent())
        ax.set_xlim([-25,10])
        ax.set_ylim([-15,15])
        ax.set_xlabel('m')
        ax.set_ylabel('m', rotation='horizontal')

        # GT path
        print((instr_id, gt[instr_id]['instructions']))
        gt_path = gt[instr_id]['path']
        gt_heading = gt[instr_id]['heading']
        print(math.degrees(gt_heading))
        gt_line, gt_start, gt_end = path_to_line(gt_path, graph)
        gt_line.set_color('blue')
        gt_line.set_marker('o')
        ax.add_line(gt_line)
        ax.scatter([gt_end[0]], [gt_end[1]], marker='*', zorder=3, color='blue', s=MARKER_SIZE, edgecolors='k')
        ax.scatter([gt_start[0]], [gt_start[1]], marker=get_arrow(gt_heading), zorder=3, color='blue', s=MARKER_SIZE, edgecolors='k')

        # Sim path
        sim_path = [v[0] for v in sim[instr_id]['trajectory']]
        sim_heading = sim[instr_id]['trajectory'][-1][1]
        sim_line, sim_start, sim_end = path_to_line(sim_path, graph, offset=(PATH_OFFSET,0))
        sim_line.set_color('darkorange')
        sim_line.set_marker('o')
        ax.add_line(sim_line)
        ax.scatter([sim_end[0]], [sim_end[1]], marker=get_arrow(sim_heading), zorder=3, color='darkorange', s=MARKER_SIZE, edgecolors='k')

        # Sim path on theta images
        sim_theta_path = [v[0] for v in sim_theta[instr_id]['trajectory']]
        sim_theta_heading = sim[instr_id]['trajectory'][-1][1]
        sim_theta_line, sim_theta_start, sim_theta_end = path_to_line(sim_theta_path, graph, offset=(0,PATH_OFFSET))
        sim_theta_line.set_color('green')
        sim_theta_line.set_marker('o')
        ax.add_line(sim_theta_line)
        ax.scatter([sim_theta_end[0]], [sim_theta_end[1]], marker=get_arrow(sim_theta_heading), zorder=3, color='green', s=MARKER_SIZE, edgecolors='k')

        # Robot path wmap
        wmap_path = robot_wmap[instr_id]['trajectory']
        wmap_line, wmap_start, wmap_end = points_to_line(wmap_path, offset=(0,PATH_OFFSET))
        start_heading = robot_wmap[instr_id]['start_heading']
        end_heading = robot_wmap[instr_id]['end_heading']
        wmap_line.set_color('red')
        ax.add_line(wmap_line)
        ax.scatter([wmap_end[0]], [wmap_end[1]], marker=get_arrow(end_heading), zorder=3, color='red', s=MARKER_SIZE, edgecolors='k', label='Start/end heading')
        ax.scatter([wmap_start[0]], [wmap_start[1]], marker=get_arrow(start_heading), zorder=3, color='red', s=MARKER_SIZE, edgecolors='k')

        # Robot path nomap
        nomap_path = robot_nomap[instr_id]['trajectory']
        nomap_line, nomap_start, nomap_end = points_to_line(nomap_path, offset=(0,PATH_OFFSET))
        start_heading = robot_nomap[instr_id]['start_heading']
        end_heading = robot_nomap[instr_id]['end_heading']
        nomap_line.set_color('cyan')
        ax.add_line(nomap_line)
        ax.scatter([nomap_end[0]], [nomap_end[1]], marker=get_arrow(end_heading), zorder=3, color='cyan', s=MARKER_SIZE, edgecolors='k', label='Start/end heading')
        ax.scatter([nomap_start[0]], [nomap_start[1]], marker=get_arrow(start_heading), zorder=3, color='cyan', s=MARKER_SIZE, edgecolors='k')

        # Legend
        legend_titles = (
          'GT path', 
          'Sim-Matt (NTDW %.2f)' % sim_scores['ndtw'][i],
          'Sim-Theta (NTDW %.2f)' % sim_theta_scores['ndtw'][i],
          'Robot w.map (NTDW %.2f)' % robot_wmap_scores['ndtw'][i],
          'Robot no map (NTDW %.2f)' % robot_nomap_scores['ndtw'][i]
        )
        ax.legend((gt_line, sim_line, sim_theta_line, wmap_line, nomap_line), legend_titles, loc='lower left', fontsize='small')
        plt.savefig('%s%s.png' % (OUT_DIR, instr_id))
        #plt.waitforbuttonpress(0) 
        plt.close()


def all_bagfiles(directory, prefix='vln-sim2real'):
    ''' Find all bag files '''
    if prefix is not None:
        entries = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix)]
    else:
        entries = [os.path.join(directory, f) for f in os.listdir(directory)]
    bagfiles = [f for f in entries if os.path.isfile(f) and f.endswith('.bag')]
    return bagfiles


def check_rosout():
    bags = all_bagfiles()
    results = []
    for filename in bags:
        bag = rosbag.Bag(filename)
        for topic, msg, t in bag.read_messages(topics=['/rosout']):
            if msg.name == '/agent':
                if msg.msg.startswith('Instr_id'):
                    results.append(msg)
    results.sort(key=lambda x: x.header.stamp.to_sec())
    for res in results:
        print(res.msg)
    print(len(results))


def evaluate(result_path):
    with open(GT) as f:
        gt_data = json.load(f)
    evaluator = SimEvaluation(gt_data, CONN)
    with open(result_path) as f:
        results = json.load(f)
    summary,scores = evaluator.score(results)
    print(summary)


if __name__ == '__main__':

    wmap_bagfiles = all_bagfiles(WMAP_BAG_DIR)
    plot_all_trajectories(wmap_bagfiles, filename='AllPaths-wmap', json_output_file=WMAP_JSON_OUTPUT)
    evaluate(WMAP_JSON_OUTPUT)    

    nomap_bagfiles = all_bagfiles(NOMAP_BAG_DIR)
    plot_all_trajectories(nomap_bagfiles, filename='AllPaths-nomap', json_output_file=NOMAP_JSON_OUTPUT)
    evaluate(NOMAP_JSON_OUTPUT) 

    plot_individual_trajectories()





