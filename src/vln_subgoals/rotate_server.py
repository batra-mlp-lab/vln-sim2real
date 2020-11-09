#!/usr/bin/env python


import numpy as np
import math


import rospy
import ros_numpy
import tf
import cv2
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image,LaserScan


class RotateServer(object):
    ''' Rotate scans and pano images into world coordinates '''

    def __init__(self):

        # Subscribe to panos and scans
        self.pano_sub = rospy.Subscriber('theta/image/decompressed', Image, self.pano_received)
        self.scan_sub = rospy.Subscriber('scan', LaserScan, self.scan_received)
        self.scans = []
        self.buffer_interval_sec = 0.1 # How many seconds between scans
        self.buffer_size = 200 # Keep up to this many scans

        # Need access to transforms to rotate scans and pano into world coordinates
        self.tf_lis = tf.TransformListener()

        # Publisher
        self.pano_pub = rospy.Publisher('/theta/image/rotated', Image, queue_size=1)
        self.scan_pub = rospy.Publisher('/scan/rotated', LaserScan, queue_size=1)
        self.bridge = CvBridge()

        rospy.loginfo('RotateServer launched')
        rospy.spin()


    def scan_received(self, scan):
        ''' Buffer scans for a period of time so they can be matched to pano '''
        if not self.scans:
            self.scans.append(scan)
        else:
            time_diff = (scan.header.stamp - self.scans[-1].header.stamp).to_sec()
            if time_diff >= self.buffer_interval_sec:
                self.scans.append(scan)
            if len(self.scans) > self.buffer_size:
                self.scans.pop(0)


    def get_scan(self, stamp):
        if not self.scans:
            return None
        ix = 0
        lowest_diff = abs((self.scans[ix].header.stamp - stamp).to_sec())
        for i,scan in enumerate(self.scans):
            diff = abs((scan.header.stamp - stamp).to_sec())
            if diff < lowest_diff:
                lowest_diff = diff
                ix = i
        return self.scans[ix],lowest_diff


    def pano_received(self, image):

        scan,time_diff = self.get_scan(image.header.stamp)

        if not scan:
            rospy.logerr('RotateServer received pano image but no laser scan available. Please switch on the laser scanner!')
            return
        if time_diff > 10.0:
            rospy.logerr('RotateServer received pano image but the laser scan is stale (timestamp diff %.1f secs). Is the laser scanner running?' % time_diff)
        else:
            rospy.logdebug('RotateServer received pano image and scan: timestamp diff %.1f secs' % time_diff)

        try:
            # Get transforms #TODO probably should give the theta camera a transform, rather than using base_footprint
            pano_trans,pano_rot = self.tf_lis.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            scan_trans,scan_rot = self.tf_lis.lookupTransform('/map', '/hokuyo_laser_frame', rospy.Time(0)) # avoid extrapolation into past error
        except Exception as e:
            rospy.logerr('RotateServer could not get transform, dropping pano: %s' % str(e))
            return

        # Calculate heading, turning right is positive (z-up)
        laser_heading_rad = euler_from_quaternion(scan_rot)[2]
        pano_heading_rad = euler_from_quaternion(pano_rot)[2]

        # Rotate the pano image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image)
            x_axis = 1
            roll_pixels = -int(pano_heading_rad / (math.pi * 2) * cv_image.shape[x_axis])
            cv_image = np.roll(cv_image, roll_pixels, axis=x_axis)
            rolled_image = self.bridge.cv2_to_imgmsg(cv_image, encoding=image.encoding)
            rolled_image.header = image.header
            self.pano_pub.publish(rolled_image)
        except CvBridgeError as e:
            rospy.logerr('RotateServer opencv error, dropping pano: %s' % str(e))
            return

        # Fill the missing part of the scan then rotate
        full_size = int(2*math.pi / scan.angle_increment) # 1440 for hokuyo
        actual_size = len(scan.ranges)
        start_pad = (full_size - actual_size)//2
        end_pad = full_size - actual_size - start_pad
        dummy_scan = np.pad(np.array(scan.ranges, dtype=np.float32), (start_pad, end_pad), 
                mode='constant', constant_values=np.nan)

        # Construct output scan
        output = scan
        output.header.frame_id = 'base_footprint' #TODO Height will not be correct, doesn't matter for us
        output.header.stamp = image.header.stamp # For synchronization by the agent
        output.angle_min = -math.pi
        output.angle_max = math.pi
        # roll_pos negated since laser actually goes from right to left! 
        roll_pos = int(laser_heading_rad / (math.pi * 2) * len(dummy_scan))
        output.ranges = np.roll(dummy_scan, roll_pos)
        self.scan_pub.publish(output)


if __name__ == '__main__':

    rospy.init_node('rotate_server', anonymous=False)
    my_node = RotateServer()


