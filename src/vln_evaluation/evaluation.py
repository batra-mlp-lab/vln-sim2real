#!/usr/bin/env python

import json
import math
import numpy as np
import networkx as nx
import yaml

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Twist
import dynamic_reconfigure.client

from vln_agent.srv import Instruction, InstructionResponse
from vln_agent.msg import InstructionResult
from visualization_msgs.msg import Marker
from std_msgs.msg import String,ColorRGBA
from std_srvs.srv import Trigger, TriggerResponse

from eval import load_nav_graph

import rospy
import tf
from tf.transformations import quaternion_from_euler


class Evaluation(object):
    ''' Evaluating the robot on the real environment! '''

    def __init__(self):

        rospy.on_shutdown(self.shutdown)

        self.load_dataset()
        self.results = []
        self.success_rate = []
        self.nav_error = []
        self.sent = []   # Sent instructions

        self.has_map = rospy.get_param("has_map", True)
        if self.has_map:
            agent_node = '/agent'
            move_base_node = '/move_base'
        else:
            agent_node = '/agent_relay'
            move_base_node = '/move_base_multimaster'
            # Configure service to relaunch the nav stack each episode
            nav_service = "nav_launcher/restart"
            rospy.loginfo('Evaluation waiting for %s' % nav_service)
            rospy.wait_for_service(nav_service)
            self.restart_nav_stack = rospy.ServiceProxy(nav_service, Trigger)
            rospy.loginfo('Evaluation connected to %s service' % nav_service)

        # Connect to the agent
        agent_serv = '%s/instruct' % agent_node
        rospy.loginfo('Evaluation waiting for %s service' % agent_serv)
        rospy.wait_for_service(agent_serv)
        self.agent_service = rospy.ServiceProxy(agent_serv, Instruction)
        rospy.loginfo('Evaluation connected to agent')

        # Subscribe to the agents trajectory result
        pano_sub = rospy.Subscriber('%s/result' % agent_node, InstructionResult, self.process_result)

        # Make sure we can find out where we are
        self.tf_lis = tf.TransformListener()

        # Connect to move_base to control setup between episodes
        rospy.loginfo("Evaluation waiting for move_base")
        self.move_base = actionlib.SimpleActionClient(move_base_node, MoveBaseAction)
        self.move_base.wait_for_server()
        rospy.loginfo("Evaluation connected to move_base")

        # To dynamic reconfigure move_base yaw tolerance
        if self.has_map:
            self.mb = dynamic_reconfigure.client.Client('%s/DWAPlannerROS' % move_base_node)

        # Publisher to manually control the robot (e.g. to stop it)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        self.init_markers()

        # Send first instructions
        rospy.sleep(5)
        self.go_send_nearest_instruction()
        rospy.spin()


    def configure_move_base(self, starting_episode=True):
        if self.has_map:
            if starting_episode:
                params = { 'yaw_goal_tolerance' : 0.3 }
            else:
                params = { 'yaw_goal_tolerance' : 3.0 }
            config = self.mb.update_configuration(params)


    def get_loc(self):
        # May need to handle exceptions?
        return self.tf_lis.lookupTransform('/map', '/base_footprint', rospy.Time(0))

    def nearest_node(self):
        ''' Get to the euclidean nearest graph node '''
        trans,rot = self.get_loc()

        min_dist = np.Inf
        viewpoint = None
        for node in nx.nodes(self.graph):
            pos = self.graph.node[node]['position']
            dist = math.sqrt((pos[0]-trans[0])**2+(pos[1]-trans[1])**2)
            if dist < min_dist:
                min_dist = dist
                viewpoint = node
        return viewpoint


    def go_send_nearest_instruction(self, cancel_on_fail=False):
        ''' Get to the nearest graph node which has an instruction '''
        if len(self.data) == 0:
            rospy.loginfo('Evaluation FINISHED!')
            return

        # Closest graph node
        viewpoint = self.nearest_node()
        # From closest graph node to each start point
        min_dist = np.Inf
        ix = -1
        for i,item in enumerate(self.data):
            start_viewpointId = item['path'][0]
            dist = self.distances[viewpoint][start_viewpointId]
            if dist < min_dist:
                ix = i
                min_dist = dist

        rospy.loginfo('Evaluation moving to start a new episode - instruction %s' % self.data[ix]['instr_id'])
        start_pos = self.graph.node[self.data[ix]['path'][0]]['position']
        start_heading = 0.5*math.pi - self.data[ix]['heading']
        goal_pos = self.graph.node[self.data[ix]['path'][-1]]['position']

        self.delete_markers()
        # This is the path we have chosen
        self.publish_markers(start_pos, goal_pos)

        # Send go command
        self.configure_move_base(starting_episode=True)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose()
        goal.target_pose.pose.position.x = start_pos[0]
        goal.target_pose.pose.position.y = start_pos[1]
        goal.target_pose.pose.position.z = 0
        goal.target_pose.pose.orientation = Quaternion()
        q = quaternion_from_euler(0, 0, start_heading)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        self.move_base.send_goal(goal)

        finished_within_time = self.move_base.wait_for_result(rospy.Duration(120)) 
        self.configure_move_base(starting_episode=False)
        if not finished_within_time:
            # If we don't get there in time, just abort the goal and try again
            self.move_base.cancel_goal()
            rospy.logwarn("Evaluation timed out trying to get the robot to the next start location, trying again...")
            self.go_send_nearest_instruction()
        else:
            state = self.move_base.get_state()
            if state == GoalStatus.SUCCEEDED:
                # We are at the start point! If nomap setting, restart nav stack
                if not self.has_map:
                    self.restart_nav_stack()
                # Send this instruction to agent
                item = self.data.pop(ix)
                self.sent.append(item)
                result = self.agent_service(item['instructions'], item['instr_id'])
                if not result.success:
                    rospy.logerr('Evaluation error sending instruction: %s' % result.message)
            else:
                rospy.logwarn("Evaluation did not reach the next start location, trying again...")
                self.go_send_nearest_instruction()


    def load_dataset(self):
        rospy.loginfo('Evaluation loading dataset')

        # Load a file indicating the done and skipped instruction ids
        instruct_params_file = rospy.get_param('instruction_params_file')
        with open(instruct_params_file) as f:
            self.instruct_params = yaml.load(f)

        # Load nav graph
        self.conn_file = rospy.get_param('connectivity_file')
        self.graph = load_nav_graph(self.conn_file)
        # compute all shortest paths
        self.paths = dict(nx.all_pairs_dijkstra_path(self.graph))
        # compute all shortest paths
        self.distances = dict(nx.all_pairs_dijkstra_path_length(self.graph))

        # Load instructions
        self.dataset_file = rospy.get_param('instruction_dataset_file')
        self.data = []
        with open(self.dataset_file) as f:
            data = json.load(f)
            for item in data:
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    new_id = '%s_%d' % (item['path_id'], j)

                    if new_id in self.instruct_params['skip'] or new_id in self.instruct_params['done']:
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = new_id
                    new_item['instructions'] = instr
                    self.data.append(new_item)
        rospy.loginfo('Loaded %d instructions for evaluation' % len(self.data))

        comp_file = rospy.get_param('comparison_results_file', '') 
        if comp_file:
            rospy.loginfo('Evaluation - found comparison results file')
            self.comp_data = {}
            with open(comp_file) as f:
                data = json.load(f)
                for item in data:
                    self.comp_data[item['instr_id']] = item
        else:
            self.comp_data = None


    def process_result(self, res):

        # Provide a rough guess of performance using euclidean distance
        instr_id = res.instr_id
        last = self.sent[-1]
        assert instr_id == last['instr_id']
        goal_viewpoint = last['path'][-1]
        goal_pos = self.graph.node[goal_viewpoint]['position']
        trans,rot = self.get_loc()
        nav_error = math.sqrt((goal_pos[0]-trans[0])**2+(goal_pos[1]-trans[1])**2)

        self.success_rate.append(nav_error < 3.0)
        self.nav_error.append(nav_error)
        if nav_error < 3.0:
            rospy.loginfo('Success! Nav error: %.2fm' % nav_error)
        else:
            rospy.loginfo('Nav error: %.2fm' % nav_error)

        if self.comp_data is not None:
            comp_final_viewpoint = self.comp_data[instr_id]['trajectory'][-1][0]
            comp_pos = self.graph.node[comp_final_viewpoint]['position']
            comp_nav_error = math.sqrt((goal_pos[0]-comp_pos[0])**2+(goal_pos[1]-comp_pos[1])**2)
            rospy.loginfo('Sim Nav error: %.2fm (%s)' % (comp_nav_error, 'Success!' if comp_nav_error < 3.0 else 'Failure'))

        N = len(self.nav_error)
        rospy.loginfo('Average Nav error: %.2fm (%d episodes)' % (np.average(np.array(self.nav_error)),N))
        rospy.loginfo('Average Success rate: %.2f (%d episodes)' % (np.average(np.array(self.success_rate)),N))

        if res.reason == 'predicted stop action':
            # Save that we've done this instruction id
            rospy.loginfo('Evaluation recording this instr_id as done: %s' % instr_id)
            self.instruct_params['done'].append(instr_id)
            instruct_params_file = rospy.get_param('instruction_params_file')
            with open(instruct_params_file, 'w') as f:
                yaml.dump(self.instruct_params, f)
        else:
            rospy.logwarn('Evaluation not recording instr as done: %s, reason: %s' % (instr_id, res.reason))

        # Send a new instruction
        self.go_send_nearest_instruction()


    def shutdown(self):
        # Cancel any active goals
        self.move_base.cancel_goal()
        rospy.sleep(2)
        # Stop the robot
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)


    def delete_markers(self):
        self.markers = Marker()
        self.markers.action = Marker.DELETEALL
        self.marker_pub.publish(self.markers)


    def init_markers(self):
        # Set up our start and goal markers
        marker_scale = 0.3
        
        # Define a marker publisher.
        self.marker_pub = rospy.Publisher('goal_markers', Marker, queue_size=10)
        
        # Initialize the marker points list.
        self.markers = Marker()
        self.markers.ns = 'goal'
        self.markers.id = -1
        self.markers.type = Marker.SPHERE_LIST
        self.markers.action = Marker.ADD
        self.markers.lifetime = rospy.Duration(0) # 0 is forever
        self.markers.scale.x = marker_scale
        self.markers.scale.y = marker_scale
        self.markers.header.frame_id = 'map'


    def publish_markers(self, start_pos, goal_pos):
        self.markers.id += 1
        self.markers.header.stamp = rospy.Time.now()
        self.markers.points = []
        self.markers.colors = []
        self.markers.points.append(Point(start_pos[0], start_pos[1], 1.35))
        self.markers.colors.append(ColorRGBA(0,1,0,1))
        self.markers.points.append(Point(goal_pos[0], goal_pos[1], 1.35))
        self.markers.colors.append(ColorRGBA(1,0.6,0,1))
        self.marker_pub.publish(self.markers)


if __name__ == '__main__':

    rospy.init_node('evaluation', anonymous=False)
    my_node = Evaluation()


