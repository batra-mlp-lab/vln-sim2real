#!/usr/bin/env python

import json
import math
import numpy as np
import networkx as nx
import torch

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *

from vln_evaluation.msg import LocationHint
from vln_agent.srv import Instruction, InstructionResponse
from vln_agent.msg import InstructionResult


import rospy
from eval import SimEvaluation

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

class MockEvaluation(object):
    ''' Evaluating the robot on the simulator environment '''

    def __init__(self):

        self.load_dataset()
        self.results = []
        self.success_rate = []
        self.nav_error = []

        # Subscribe to the agents trajectory result
        pano_sub = rospy.Subscriber('agent/result', InstructionResult, self.process_result)

        # Prepare to send location hints
        self.pub = rospy.Publisher('mock/hint', LocationHint, queue_size=1)

        # Connect to the agent
        agent_serv = 'agent/instruct'
        rospy.loginfo('Mock evaluation waiting for %s service' % agent_serv)
        rospy.wait_for_service(agent_serv)
        self.agent_service = rospy.ServiceProxy(agent_serv, Instruction)
        rospy.loginfo('Mock evaluation connected to agent')

        # We need to provide move base as well!
        self.move_base = actionlib.SimpleActionServer('/move_base', MoveBaseAction, 
                              execute_cb=self.move_base, auto_start=False)
        self.move_base.start()

        # Send first location hint and instructions
        rospy.sleep(5)
        self.send_next_starting_location_hint()
        self.send_next_instruction()
        rospy.spin()


    def move_base(self, goal):
        rospy.logdebug('Mock evaluation moving to goal')

        # Now to look up the goal in the navigation graph, send the new hint, say that it succeeded.
        target_pos = goal.target_pose.pose.position
        min_dist = np.Inf
        viewpoint = None
        heading = 0
        for node in nx.nodes(self.eval.graph):
            pos = self.eval.graph.node[node]['position']
            dist = math.sqrt((pos[0]-target_pos.x)**2+(pos[1]-target_pos.y)**2)
            if dist < min_dist:
                min_dist = dist
                viewpoint = node
        assert(min_dist < 0.1)

        # Set hint
        hint = LocationHint()
        hint.viewpoint = viewpoint
        hint.ros_heading =  math.atan2(target_pos.y-self.curr_y, target_pos.x-self.curr_x)
        self.pub.publish(hint)
        self.curr_x = target_pos.x
        self.curr_y = target_pos.y
        rospy.logdebug('Mock evaluation position (%.2f, %.2f), sending hint: %s' % 
                                        (self.curr_x, self.curr_y, str(hint)))
        self.move_base.set_succeeded(goal)


    def send_next_starting_location_hint(self):
        hint = LocationHint()
        hint.viewpoint = self.data[self.ix]['path'][0]
        hint.ros_heading = 0.5*math.pi - self.data[self.ix]['heading']
        # initialize location to calculate as the basis for next heading
        pos = self.eval.graph.node[hint.viewpoint]['position']
        self.curr_x = pos[0]
        self.curr_y = pos[1]
        rospy.logdebug('Mock evaluation initial position (%.2f, %.2f), sending hint: %s' %
                                       (self.curr_x, self.curr_y, str(hint)))
        self.pub.publish(hint)


    def send_next_instruction(self):
        item = self.data[self.ix]
        result = self.agent_service(item['instructions'], item['instr_id'])
        if not result.success:
            rospy.logerr('Mock evaluation error sending instruction: %s' % result.message)


    def load_dataset(self):
        rospy.loginfo('Mock evaluation loading dataset')
        self.dataset_file = rospy.get_param('instruction_dataset_file')
        self.conn_file = rospy.get_param('connectivity_file')
        self.data = []
        with open(self.dataset_file) as f:
            data = json.load(f)
            self.eval = SimEvaluation(data, self.conn_file)
            for item in data:
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    # Hack to start with a known instruction id (for debugging)
                    if new_item['instr_id'] == '11_0':
                        self.data.insert(0,new_item)
                    else:
                        self.data.append(new_item)
        self.ix = 0
        comp_file = rospy.get_param('comparison_results_file', '') 
        if comp_file:
            self.comp_data = {}
            with open(comp_file) as f:
                data = json.load(f)
                for item in data:
                    self.comp_data[item['instr_id']] = item
        else:
            self.comp_data = None


    def process_result(self, res):
        item = {
            'instr_id': res.instr_id,
            # In this case image_filenames will contain the viewpoints
            'trajectory': [(viewpt,0,0) for viewpt in res.image_filenames]
        }
        self.results.append(item)
        nav_error = self.eval.nav_error(item['instr_id'], item['trajectory'])
        self.success_rate.append(nav_error < 3.0)
        self.nav_error.append(nav_error)
        if nav_error < 3.0:
            rospy.loginfo('Success! Nav error: %.2fm' % nav_error)
        else:
            rospy.loginfo('Nav error: %.2fm' % nav_error)
        if self.comp_data is not None:
            comp_traj = []
            for v in self.comp_data[item['instr_id']]['trajectory']:
                if len(comp_traj) == 0 or v[0] != comp_traj[-1]:
                    comp_traj.append(v[0])
            our_traj = [v[0] for v in item['trajectory']]
            rospy.loginfo('Ours: %s' % str(our_traj))
            rospy.loginfo('Theirs: %s' % str(comp_traj))

        N = len(self.nav_error)
        rospy.loginfo('Average Nav error: %.2fm (%d episodes)' % (np.average(np.array(self.nav_error)),N))
        rospy.loginfo('Average Success rate: %.2f (%d episodes)' % (np.average(np.array(self.success_rate)),N))

        output = rospy.get_param('results_output_file')
        rospy.loginfo('Saving output to %s' % output)
        with open(output, 'w') as f:
            json.dump(self.results, f)

        if len(self.results) == len(self.data):
            score_summary, scores = self.eval.score(self.results)
            rospy.loginfo('Mock evaluation scores:\n %s' % str(score_summary))
            score_output = rospy.get_param('scores_output_file')
            rospy.loginfo('Saving scores to %s' % score_output)
            with open(score_output, 'w') as f:
                json.dump(scores, f)
        else:
            self.ix += 1
            if self.ix < len(self.data):
                self.send_next_starting_location_hint()
                self.send_next_instruction()


if __name__ == '__main__':

    rospy.init_node('mock_evaluation', anonymous=False)
    my_node = MockEvaluation()


