#!/usr/bin/env python

import numpy as np
import math
import json
import networkx as nx

import rospy
import ros_numpy
import tf
from sensor_msgs.msg import Image,LaserScan
from geometry_msgs.msg import PoseArray,Pose,Quaternion,Point

from subgoal_server import CNN,euler_to_quaternion,quaternion_to_euler
from vln_evaluation.msg import LocationHint
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


def load_nav_graph(conn_file):
    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    with open(conn_file) as f:
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
    return G


def heading_diff_rad(h1, h2):
    assert h1 >= 0 and h1 <= math.pi*2 and h2 >= 0 and h2 <= math.pi*2, (h1,h2)
    return min(abs(h1-h2), 2*math.pi-h1+h2, 2*math.pi+h1-h2)



class SubgoalMockServer(object):

    def __init__(self):

        self.cnn = CNN()

        if False:
            # Load features
            import csv
            import base64
            import sys
            csv.field_size_limit(sys.maxsize)
            TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
            VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
            FEATURE_SIZE = 2048
            infile = '/root/mount/vln-pano2sim-ros/src/vln_subgoals/models/ResNet-152-imagenet-pytorch-ported-coda.tsv'
            self.fake_data = {}
            with open(infile, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
                for item in reader:
                    if item['scanId'] == 'yZVvKaJZghh':
                        self.fake_data[item['viewpointId']] = np.frombuffer(base64.b64decode(item['features']),
                            dtype=np.float32).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
            print('loaded features')

        # Make sure we can find out where we are
        self.tf_lis = tf.TransformListener()

        # Subscribe to panos rotated to the world frame - x axis down center of the image
        pano_sub = rospy.Subscriber('theta/image/rotated', Image, self.predict_waypoints)

        # Load the navigation graph
        self.connectivity_file = rospy.get_param('connectivity_file')
        self.graph = load_nav_graph(self.connectivity_file)

        # Subscribe to hints about the current location
        self.sub = rospy.Subscriber('mock/hint', LocationHint, self.save_location)
        self.viewpoint = None

        # Publisher
        self.pub_feat = rospy.Publisher('subgoal/features', Image, queue_size=1)
        self.pub_way = rospy.Publisher('subgoal/waypoints', PoseArray, queue_size=1)

        # Define a marker publisher.
        self.marker_pub = rospy.Publisher('nav_graph_nodes', Marker, queue_size=10)

        rospy.spin()


    def publish_nav_graph_markers(self):
        # Set up our waypoint markers
        marker_scale = 0.2
        
        # Initialize the marker points list.
        markers = Marker()
        markers.ns = 'waypoints'
        markers.id = 0
        markers.type = Marker.SPHERE_LIST
        markers.action = Marker.ADD
        markers.lifetime = rospy.Duration(0) # 0 is forever
        markers.scale.x = marker_scale
        markers.scale.y = marker_scale
        markers.scale.z = marker_scale
        markers.color.r = 0.0
        markers.color.g = 0.0
        markers.color.b = 1.0
        markers.color.a = 1.0
        markers.header.frame_id = 'map'
        markers.header.stamp = rospy.Time.now()
        markers.points = []

        for viewpoint in self.graph:
            position = self.graph.node[viewpoint]['position']
            pos = Point(position[0], position[1], marker_scale)
            markers.points.append(pos)
        self.marker_pub.publish(markers)


    def nearest_node(self):
        ''' Get to the nearest graph node which has an instruction '''
        is_sim = rospy.get_param('vln_simulation', False)

        if is_sim:
            if self.viewpoint is None:
                rospy.logerr('Subgoal mock server expected a hint, but didn\'t receive it. Don\'t know location!')
            return self.viewpoint,self.ros_heading

        else:
            # May need to handle exceptions?
            trans,rot = self.tf_lis.lookupTransform('/map', '/base_footprint', rospy.Time(0))

            min_dist = np.Inf
            viewpoint = None
            for node in nx.nodes(self.graph):
                pos = self.graph.node[node]['position']
                dist = math.sqrt((pos[0]-trans[0])**2+(pos[1]-trans[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    viewpoint = node

            r,p,heading = quaternion_to_euler(rot)
            return viewpoint,heading


    def save_location(self, hint):
        self.viewpoint = hint.viewpoint
        self.ros_heading = hint.ros_heading


    def predict_waypoints(self, image):
        rospy.loginfo('Subgoal mock server predicting waypoints')
        self.publish_nav_graph_markers()

        self.viewpoint,self.ros_heading = self.nearest_node()
        stamp = rospy.Time.now()

        # Extract cnn features plus their heading and elevation
        feats, imgs_he = self.cnn.extract_features(image)

        if False:
            #FAKE DATA FROM TSV
            import torch
            feats = torch.from_numpy(self.fake_data[self.viewpoint].transpose())
        aug_feats = np.concatenate((feats.cpu().numpy(), imgs_he), axis=0)

        # Get waypoints from nav graph and
        # Extract candidate features - each should be the closest to that viewpoint
        # Note: The candidate_feat at last position is the feature for the END stop signal (zeros)
        feat_dim = feats.shape[0]
        num_candidates = len(self.graph[self.viewpoint])+1
        candidates = np.zeros((feat_dim+2, num_candidates), dtype=np.float32)
        im_features = feats.cpu().numpy().reshape(feat_dim, 3, 12)
        imgs_he = imgs_he.reshape(2, 3, 12)

        # Publish pose array of possible goals
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = 'map'

        curr_pos = self.graph.node[self.viewpoint]['position']
        for i,waypoint_id in enumerate(self.graph[self.viewpoint]):
            waypoint_pos = self.graph.node[waypoint_id]['position']
            target_rel = waypoint_pos - curr_pos
            # Target heading in ros coordinates
            target_heading = math.atan2(target_rel[1], target_rel[0])
            if target_heading < 0: 
                target_heading += 2*math.pi
            # match heading to a feature in the image
            min_diff = 2*math.pi
            for ix,heading in enumerate(imgs_he[0,1]):
                # rely both inputs have same 0 to 2pi range
                heading_diff = heading_diff_rad(heading,target_heading)
                if heading_diff < min_diff:
                    min_diff = heading_diff
                    heading_ix = ix
            candidates[:-2, i] = im_features[:, 1, heading_ix]
            candidates[-2:, i] = [target_heading, 0]   # heading, elevation

            # Construct pose output as well
            pose = Pose()
            pose.position.x = waypoint_pos[0]
            pose.position.y = waypoint_pos[1]
            pose.position.z = 0
            pose.orientation = euler_to_quaternion(0, 0, target_heading)
            pa.poses.append(pose)

        # Publish image and candidate features
        combined_feats = np.concatenate([aug_feats,candidates], axis=1)
        np_feats = ros_numpy.msgify(Image, combined_feats, encoding="32FC1")
        np_feats.header.stamp = stamp
        self.pub_feat.publish(np_feats)
        rospy.logdebug('Subgoal mock server published features')

        # Put current pose in last position to feed the agent_server
        pose = Pose()
        pose.position.x = curr_pos[0]
        pose.position.y = curr_pos[1]
        pose.position.z = 0
        pose.orientation = euler_to_quaternion(0, 0, self.ros_heading)
        pa.poses.append(pose)
        self.pub_way.publish(pa)
        self.viewpoint = None
        rospy.logdebug('Subgoal mock server published waypoints')


if __name__ == '__main__':

    rospy.init_node('subgoal_mock_server', anonymous=False)
    my_node = SubgoalMockServer()


