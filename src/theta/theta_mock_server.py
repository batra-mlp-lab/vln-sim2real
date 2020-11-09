#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from PIL import Image as Img
from vln_evaluation.msg import LocationHint
import numpy as np


class ThetaMockServer(object):
    """ Mock camera server, subscribes to mock/hint topic that says which node it is at,
        and then the camera publishes a pano image from the file system """

    def __init__(self):

        # Fire up the camera
        rospy.loginfo('Detected mock camera')

        # Service
        self.service = rospy.Service('theta/capture', Trigger, self.capture)

        # By default we publish directly to image/rotated since all images are already aligned to the world frame
        self.pub_image = rospy.Publisher(rospy.get_param('theta_topic', 'theta/image/rotated'), Image, queue_size=1)

        # Extra - subscribe to the location, viewpoint etc.
        self.data_dir = rospy.get_param('pano_images_dir')
        self.sub = rospy.Subscriber('mock/hint', LocationHint, self.next_image)
        self.next_path = None

        rospy.spin()


    def next_image(self, data):
        self.viewpoint = data.viewpoint
        self.next_path = self.data_dir + '/' + data.viewpoint + '_equirectangular.jpg'


    def capture(self, req):
        rospy.logdebug('Capturing mock panorama')

        if not self.next_path:
            msg = 'Theta mock server did not receive a viewpoint hint.'
            return TriggerResponse(success=False, message=msg)

        img = Img.open(self.next_path)
        # This only works for coda, where all panos have the y-axis in the center of the 
        # image instead of the x. So roll by -90 degrees to x-axis is in the middle of image
        #np_img = np.array(img)
        #np_img = np.roll(np_img, -np_img.shape[1]//4, axis=1) 
        #img = Img.fromarray(np_img)
        rospy.logdebug('Mock panorama captured!')

        image = Image(height=img.height, width=img.width, encoding="rgb8", is_bigendian=False, step=img.width*3, data=img.tobytes())
        image.header.stamp = rospy.Time.now()
        image.header.frame_id = 'map'

        self.pub_image.publish(image)
        rospy.logdebug('Mock panorama published!')

        self.next_path = None

        # Put the viewpoint id here because it makes mock evaluation easy
        return TriggerResponse(success=True, message=self.viewpoint)




if __name__ == '__main__':

    rospy.init_node('theta_mock', anonymous=False)
    my_node = ThetaMockServer()


