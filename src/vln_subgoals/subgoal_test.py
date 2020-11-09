#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger, TriggerResponse


class SubgoalTest(object):
    """ Simple node to periodically trigger the mock theta camera
        to test the subgoal model. """

    def __init__(self):

        rospy.loginfo('Starting subgoal test')

        self.cam_service = rospy.ServiceProxy('theta/capture', Trigger)
        rate = rospy.Rate(10) # 10hz

        while not rospy.is_shutdown():
            result = self.cam_service()
            rate.sleep()


if __name__ == '__main__':

    rospy.init_node('subgoal_test', anonymous=False)
    my_node = SubgoalTest()


