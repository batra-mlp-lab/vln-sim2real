#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from itertools import izip_longest
from cnn import PyTorchCNN as CNN

import rospy
import ros_numpy
import message_filters
import tf
from sensor_msgs.msg import Image,LaserScan
from geometry_msgs.msg import PoseArray,Pose,Quaternion
from unet import UNet


def quaternion_to_euler(q):

    try:
        x=q.x; y=q.y; z=q.z; w=q.w
    except:
        x=q[0]; y=q[1]; z=q[2]; w=q[3]

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

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return Quaternion(qx, qy, qz, qw)


def neighborhoods(mu, x_range, y_range, sigma, circular_x=True, gaussian=False):
    """ Generate masks centered at mu of the given x and y range with the
        origin in the centre of the output 
    Inputs:
        mu: tensor (N, 2)
    Outputs:
        tensor (N, y_range, s_range)
    """
    x_mu = mu[:,0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:,1].unsqueeze(1).unsqueeze(1)
    # Generate bivariate Gaussians centered at position mu
    x = torch.arange(start=0,end=x_range, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
    y = torch.arange(start=0,end=y_range, device=mu.device, dtype=mu.dtype).unsqueeze(1).unsqueeze(0)
    y_diff = y - y_mu
    x_diff = x - x_mu
    if circular_x:
        x_diff = torch.min(torch.abs(x_diff), torch.abs(x_diff + x_range))
    if gaussian:
        output = torch.exp(-0.5 * ((x_diff/sigma)**2 + (y_diff/sigma)**2 ))
    else:
        output = 0.5*(torch.abs(x_diff) <= sigma).type(mu.dtype) + 0.5*(torch.abs(y_diff) <= sigma).type(mu.dtype)
        output[output < 1] = 0
    return output


def nms(pred, sigma, thresh, max_predictions, gaussian=False):
    ''' Input (batch_size, 1, height, width) '''

    shape = pred.shape
    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0],-1))
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0],-1))
    for i in range(max_predictions):
        # Find and save max
        flat_supp_pred = supp_pred.reshape((shape[0],-1))
        val,ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0,shape[0])
        flat_output[indices,ix] = flat_pred[indices,ix]

        # Suppression
        y = ix / shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x,y], dim=1).float()
        g = neighborhoods(mu, shape[-1], shape[-2], sigma, gaussian=gaussian)
        supp_pred *= (1-g.unsqueeze(1))

    # Make sure you always have at least one detection
    output[output < min(thresh,output.max())] = 0
    return output


class SubgoalServer(object):

    def load_subgoal_model(self):
        ''' Load the pretrained model to predict nearby waypoints from 
            pano and laser scan data '''
        self.unet_weights = rospy.get_param('unet_weights_file')
        self.unet = UNet(n_channels=2, n_classes=1).to(self.device)
        self.unet.load_state_dict(torch.load(self.unet_weights, map_location=self.device))
        self.unet.eval()
        rospy.loginfo('Subgoal model loaded weights')


    def __init__(self):

        # Fire up some networks
        self.device = torch.device(rospy.get_param('device', 'cuda:0'))
        self.load_subgoal_model()
        self.cnn = CNN()

        # Make sure we can find out where we are
        self.tf_lis = tf.TransformListener()

        # Subscribe to panos and scans that have been processed by rotate_pano
        pano_sub = message_filters.Subscriber('theta/image/rotated', Image)
        scan_sub = message_filters.Subscriber('scan/rotated', LaserScan)
        ts = message_filters.TimeSynchronizer([pano_sub, scan_sub], 1)
        ts.registerCallback(self.predict_waypoints)

        # NMS param
        self.max_predictions=rospy.get_param('max_subgoal_predictions',10)

        # Publisher
        self.pub_feat = rospy.Publisher('subgoal/features', Image, queue_size=1)
        self.pub_occ = rospy.Publisher('subgoal/occupancy', Image, queue_size=1)
        self.pub_prob = rospy.Publisher('subgoal/prob', Image, queue_size=1)
        self.pub_nms = rospy.Publisher('subgoal/nms_prob', Image, queue_size=1)
        self.pub_way = rospy.Publisher('subgoal/waypoints', PoseArray, queue_size=1)
        rospy.spin()


    def prep_scan_for_net(self, scan_img):
        imgs = np.empty((1, 2, scan_img.shape[0], scan_img.shape[1]), dtype=np.float32)
        imgs[:, 1, :, :] = scan_img.transpose((2,0,1))
        ran_ch = np.linspace(-0.5, 0.5, num=imgs.shape[2])
        imgs[:, 0, :, :] = np.expand_dims(np.expand_dims(ran_ch, axis=0), axis=2)
        out = torch.from_numpy(imgs).to(device=self.device)
        return out


    def radial_occupancy(self, scan):
        ''' Convert an 1D numpy array of 360 degree range scans to a 2D numpy array representing
            a radial occupancy map. Values are 1: occupied, -1: free, 0: unknown 
            Here we assume the scan is a full 360 degrees due to preprocessing by rotate_pano.'''
        n_range_bins = rospy.get_param('range_bins')
        n_heading_bins = rospy.get_param('heading_bins')
        range_bin_width = rospy.get_param('range_bin_width')
        range_bins = np.arange(0, range_bin_width*(n_range_bins+1), range_bin_width)
        heading_bin_width = 360.0/n_heading_bins

        # Record the heading, range of the center of each bin in ros coords. Heading increases as you turn left.
        hr = np.zeros((n_range_bins, n_heading_bins, 2), dtype=np.float32)
        range_centers = range_bins[:-1]+range_bin_width/2
        hr[:,:,1] = range_centers.reshape(-1,1)
        assert n_heading_bins % 2 == 0
        heading_centers = -(np.arange(n_heading_bins)*heading_bin_width+heading_bin_width/2-180)
        hr[:,:,0] = np.radians(heading_centers)

        output = np.zeros((n_range_bins, n_heading_bins, 1), dtype=np.float32) # rows, cols, channels
        # chunk scan data to generate occupied (value 1)
        chunk_size = len(scan.ranges)//n_heading_bins
        args = [iter(scan.ranges[::-1])] * chunk_size # reverse scan since it's from right to left!
        n = 0
        for chunk in izip_longest(*args):
            # occupied (value 1)
            chunk = np.array(chunk)
            chunk[np.isnan(chunk)]=-1 # Remove nan values, negatives will fall outside range_bins
            # Add 'inf' as right edge of an extra bin to account for the case if the returned range exceeds
            # the maximum discretized range. In this case we still want to register these cells as free. 
            hist, _ = np.histogram(chunk, bins=np.array(range_bins.tolist() + [np.Inf]))
            output[:,n,0] = np.clip(hist[:-1], 0, 1)
            # free (value -1)
            free_ix = np.flip(np.cumsum(np.flip(hist,axis=0), axis=0), axis=0)[1:] > 0
            output[:,n,0][free_ix] = -1
            n+=1
        return output, hr


    def predict_waypoints(self, image, scan):
        rospy.loginfo('Subgoal model predicting waypoints')
        stamp = rospy.Time.now()

        # Extract cnn features plus their heading and elevation
        feats, imgs_he = self.cnn.extract_features(image)
        aug_feats = np.concatenate((feats.cpu().numpy(), imgs_he), axis=0)
        feat_dim = feats.shape[0]

        # Prepare the scan data
        scan_img, scan_hr = self.radial_occupancy(scan)
        if rospy.get_param('subgoal_publish_occupancy', False):
            np_scan_img = ros_numpy.msgify(Image, (127*(scan_img+1)).astype(np.uint8), encoding='mono8')
            np_scan_img.header.stamp = stamp
            self.pub_occ.publish(np_scan_img)
        # Roll the scans so to match the image features
        roll_ix = -scan_img.shape[1]//4 + 2  # -90 degrees plus 2 bins
        rolled_scan_img = np.roll(scan_img, roll_ix, axis=1)
        rolled_scan_hr = np.roll(scan_hr, roll_ix, axis=1)

        # Predict subgoals
        scans = self.prep_scan_for_net(rolled_scan_img)
        feats = feats.reshape((1,feat_dim,3,12))
        with torch.no_grad():
            logits = self.unet(scans,feats)
            pred = F.softmax(logits.flatten(1), dim=1).reshape(logits.shape)

        if rospy.get_param('subgoal_publish_prob', False):
            viz_pred = np.roll(255*pred.squeeze().cpu().numpy(), -roll_ix, axis=1)
            np_viz_pred = ros_numpy.msgify(Image, viz_pred, encoding="32FC1")
            np_viz_pred.header.stamp = stamp
            self.pub_prob.publish(np_viz_pred)
        sigma = rospy.get_param('subgoal_nms_sigma')
        thresh = rospy.get_param('subgoal_nms_thresh')
        nms_pred = nms(pred, sigma, thresh, self.max_predictions)
        if rospy.get_param('subgoal_publish_nms_prob', False):
            viz_nms_pred = np.roll(255*nms_pred.squeeze().cpu().numpy(), -roll_ix, axis=1)
            np_viz_nms_pred = ros_numpy.msgify(Image, viz_nms_pred, encoding="32FC1")
            np_viz_nms_pred.header.stamp = stamp
            self.pub_nms.publish(np_viz_nms_pred)

        # Get agent pose
        trans,rot = self.tf_lis.lookupTransform('/map', '/base_footprint', rospy.Time(0))
        r,p,agent_heading_rad = quaternion_to_euler(rot) # ros heading

        # Extract waypoint candidates
        nms_pred = nms_pred.squeeze()
        waypoint_ix = (nms_pred > 0).nonzero()

        # Extract candidate features - each should be the closest to that viewpoint
        # Note: The candidate_feat at last position is the feature for the END stop signal (zeros)
        num_candidates = waypoint_ix.shape[0]+1
        candidates = np.zeros((feat_dim+2, num_candidates), dtype=np.float32)
        im_features = aug_feats[:-2].reshape(feat_dim, 3, 12)
        imgs_he = imgs_he.reshape(2, 3, 12)

        # Publish pose array of possible goals
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = 'map'

        for i,(range_bin,heading_bin) in enumerate(waypoint_ix.cpu().numpy()):
            hr = rolled_scan_hr[range_bin,heading_bin]
            # Calculate elevation to the candidate pose is 0 for the robot (stays the same height, doesn't go on stairs)
            # So candidate is always from the centre row of images 3 * 12 images
            img_heading_bin = heading_bin//4
            candidates[:-2, i] = im_features[:, 1, img_heading_bin] # 1 is for elevation 0
            candidates[-2:, i] = [hr[0], 0]   # heading, elevation

            # Construct pose output as well
            pose = Pose()
            pose.position.x = trans[0] + math.cos(hr[0])*hr[1]
            pose.position.y = trans[1] + math.sin(hr[0])*hr[1]
            pose.position.z = 0
            # Which way should the robot face when it arrives? Away from here I guess.
            candidate_heading = math.atan2(pose.position.y-trans[1], pose.position.x-trans[0])
            pose.orientation = euler_to_quaternion(0, 0, candidate_heading)
            pa.poses.append(pose)

        # Publish image and candidate features
        combined_feats = np.concatenate([aug_feats,candidates], axis=1)
        np_feats = ros_numpy.msgify(Image, combined_feats, encoding="32FC1")
        np_feats.header.stamp = stamp
        self.pub_feat.publish(np_feats)
        rospy.logdebug('Subgoal server published features')

        # Put current pose in last position to feed the agent_server
        pose = Pose()
        pose.position.x = trans[0]
        pose.position.y = trans[1]
        pose.position.z = 0
        pose.orientation = euler_to_quaternion(0, 0, agent_heading_rad)
        pa.poses.append(pose)
        self.pub_way.publish(pa)
        rospy.loginfo('Subgoal server published waypoints')


if __name__ == '__main__':

    rospy.init_node('subgoal_server', anonymous=False)
    my_node = SubgoalServer()


