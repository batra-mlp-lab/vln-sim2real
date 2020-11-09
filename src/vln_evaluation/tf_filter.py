#!/usr/bin/env python



import rospy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Transform


class TfFilter(object):
    ''' Republishes all tf messages except transform [odom] with parent [map]
        (avoids tf multiple authority contention when running two nav stacks). '''

    def __init__(self):
        rospy.loginfo('Started tf_filter')

        self.pub = rospy.Publisher('tf_gt', TFMessage, queue_size=500)

        sub = rospy.Subscriber('tf', TFMessage, self.process_transform)

        rospy.spin()


    def process_transform(self, msg):
        output = TFMessage()
        for tf in msg.transforms:
            if tf.header.frame_id == 'map' and tf.child_frame_id == 'odom':
                continue
            else:
                output.transforms.append(tf)
        if output.transforms:
            self.pub.publish(output)


if __name__ == '__main__':

    rospy.init_node('tf_filter', anonymous=False)
    my_node = TfFilter()


