#!/usr/bin/env python


from std_srvs.srv import Trigger, TriggerResponse
import rospy
import subprocess
import rosnode


class NavLauncher(object):
    ''' Service for restarting the gmapping and move_base nodes for the
        hardest setting of VLN evaluation - ensuring an empty map at the 
        start of each episode. '''

    def __init__(self):
        rospy.loginfo('Started nav_launcher')
        launch_service = rospy.Service('nav_launcher/restart', Trigger, self.relaunch)
        rospy.spin()


    def relaunch(self, req):
        killed,failed = rosnode.kill_nodes(['move_base', 'slam_gmapping'])
        result = 'move_base' in killed and 'slam_gmapping' in killed
        if not result:
            rospy.logwarn('NavLauncher failed to kill %s' % failed)
        process = subprocess.Popen(['roslaunch', 'vln_agent', 'nav_stack.launch'])
        return TriggerResponse(success=result)



if __name__ == '__main__':

    rospy.init_node('nav_launcher', anonymous=False)
    my_node = NavLauncher()


