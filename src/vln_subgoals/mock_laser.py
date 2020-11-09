#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from vln_evaluation.msg import LocationHint
import json
import math
import numpy as np


class MockLaser(object):
    """ Mock laser scanner, subscribes to mock/hint topic that says which node it is at,
        and then publishes a scan from the file system """

    def __init__(self):

        rospy.loginfo('Detected mock laser scanner')

        self.pub_scan = rospy.Publisher(rospy.get_param('scan_topic', 'scan'), LaserScan, queue_size=50)

        with open(rospy.get_param('mock_scans_path')) as jf:
            json_data = json.load(jf)
            # Build lookup by id
            self.scans = {}
            for item in json_data:
                self.scans[item['image_id']] = item

        # Subscribe to location hint
        self.viewpoint = None
        self.sub = rospy.Subscriber('mock/hint', LocationHint, self.next_viewpoint)

        rate = rospy.Rate(50) # 50hz
        while not rospy.is_shutdown():
            self.publish_scan()
            rate.sleep()

    def next_viewpoint(self, data):
        self.viewpoint = data.viewpoint
        rospy.logdebug('Received viewpoint hint: %s' % self.viewpoint)

    def publish_scan(self):
        if self.viewpoint == None:
            rospy.logdebug('Mock laser did not receive a hint to indicate the viewpoint. Not publishing mock scans!')
            return
        rospy.logdebug('Publishing mock laser scan viewpoint %s' % self.viewpoint)

        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.header.frame_id = 'hokuyo_laser_frame'

        # reverse scan since it's from right to left!
        scan.ranges = self.scans[self.viewpoint]['laser'][::-1]
        scan.intensities = [1] * len(scan.ranges)
        # Here, we are giving the full 360 scan, nothing is dropped so set the range.
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = 2*math.pi / len(scan.ranges)
        scan.time_increment = 0.01 / len(scan.ranges)
        scan.range_min = 0.0
        scan.range_max = 100.0

        self.pub_scan.publish(scan)


if __name__ == '__main__':

    rospy.init_node('mock_laser', anonymous=False)
    my_node = MockLaser()


