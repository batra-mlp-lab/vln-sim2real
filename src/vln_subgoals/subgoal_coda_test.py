#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import PoseArray
from vln_evaluation.msg import LocationHint


class SubgoalCodaTest(object):
    """ Simple node to periodically trigger the mock theta camera
        to test the subgoal model. """

    def __init__(self):

        rospy.loginfo('Starting subgoal test')

        # Coda image ids
        self.ix = 0
        self.image_ids = ['371f140f46d14d76a86930a604c8ca5d', '9dbb512f22f74f4e8ac21097a3cdf062', '7ba5ba5a95ec4b4ab90b4299bd20be0d', '380fa66464874c3fa118b5d0a62ec0cf', 'e66e18ce7b21409faccba652b07fecb3', 'd3b9a732e5914ae2bb559df7c4df953f', '3a4c287c9df7453c8cb2f073bde28c7b', '4617e18c028d4a4fb88ff77718d3c4c1', '956b425ff3134b4b940988c1e500a10b', '46d5fce9e9e14c9192bf0b4716a45cc0', '141da1f924654dca9f956f4b861ad7a3', 'd6a4f112ff4c4e339f2d961edcc25eb7', '18567b14f1fc4a10aa91c0013e968e60', '955f2e25500c4c82b0d1c8f2688a12b7', 'c0430bd6d2f74f23b65c86b79d039fcb', 'b1ea3134c8f34faf9829c77e0ae83b22', '019c5488f6a24d018b4eda5d52c8cadd', 'd58bf8c7f0a540bca0d414fd2490ff3f', '08096419df1c484b904b33af09eb441d', '80f46ddd7b964793bfd4c84084fcc45f', '7dad876929e54aa8a8c4791f882ba2d8', '784f8fe63ee741e0b83a45dd82998b3d', 'f9a5d98d13d54212b2b1b5aea4a2df98', 'd532eda1c3c2414e996db3e605edd326', '54ebdac28d5e44daa8b38111f67398a4', 'f8a0eb2ff393417b8cb2fde3cb2c67d7', '058ca22ff70349ca93779d54a11b2c21', '4f6b5a2c69b04f1ca5389f1a939a5ea4', '58cac27140844d7cbffd4bda2bf0cd97', '0f1769c8158a4eee897b4b9480189465', '1349dbc7964045e1b0af7c6d4080d180', '5cc6ea6b2ae444ad81d2b77fafce99e2', 'd3753b4ca72348338af3614d0d1b3490', 'f72c7f77a4424e1f8754b43efc75cf33', 'b407fcc6d5a942b68a0c0fa9bb15e114', 'c071869369ed4c95b9cc31b431b2a6b8', 'f742655c7c264f17abaa698777f56aeb', '15e14df42c094fb1865126a504550311', 'c52bfdcbbdb1415a9d3fabbc8fe50129', 'b1914edca2984f0a91c56de9159f948d', '8b881f7c4e0046ce81633f99295206fe', '82934363cde84139824a100e802a0448', 'd26708cf93614a1784771976ebe95ebb', '87af113f03924c158f4a96c79bef8060', '4819215c1cf94286bad53ff9ce77cc40', '3805c41871324915863943ea5287c67d', 'ba2b14dd500b4dbea33bcbd13f910ddc', '829504c3c0584485bd923bf0610993c3', '5ae769c576e44293955c1299ea694dc2', '79da6d88e55741178e4f96c25686a857', '0ae724b3780d4635ac1cf8c5a7d78cf1', '05b79c98c56342f18faec381cfb1953a', '23a6f24c5fbf40a08f75505ce53afae2', '503afa20f81b4dbc8b714e5ebfe1de8f', 'd3544b5e282d47a185157c86a599be9a', '3ab544cf180c4aa2bb010a8d9c061684', 'f8a8df104bd34f77b9c58798d4cc9440', '76c4cedb68b04231bc3511770fda3abc', 'd6845647288a46bd82c6078f81bc2dbb']

      
        self.cam_service = rospy.ServiceProxy('theta/capture', Trigger)
        self.pub = rospy.Publisher('mock/hint', LocationHint, queue_size=1)
        self.sub = rospy.Subscriber('/subgoal/waypoints', PoseArray, self.next_hint)
        
        rospy.sleep(10.) # Wait for everything to spool up
        self.next_hint(None)
        rospy.spin()

    def next_hint(self, data):
        rospy.sleep(2.) # Wait to inspect it in rviz

        # Publish hint
        hint = LocationHint()
        hint.viewpoint = self.image_ids[self.ix]
        rospy.loginfo('Testing %d/%d: %s' % (self.ix, len(self.image_ids), hint.viewpoint))
        self.pub.publish(hint)  
        self.ix += 1
        if self.ix >= len(self.image_ids):
            self.ix = 0

        # Short wait then trigger mock image capture    
        rospy.sleep(0.1)
        result = self.cam_service()


if __name__ == '__main__':

    rospy.init_node('subgoal_coda_test', anonymous=False)
    my_node = SubgoalCodaTest()


