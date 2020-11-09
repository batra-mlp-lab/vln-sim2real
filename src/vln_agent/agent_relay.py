#!/usr/bin/env python


from std_srvs.srv import Trigger, TriggerResponse
from vln_agent.srv import Instruction, InstructionResponse
from vln_agent.msg import InstructionResult

import rospy



class AgentRelay(object):
    ''' Relay service for sending instructions to the agent running on PC in the hardest
        evaluation setting. Overcomes the issue that multimaster doesn't appear to sync
        the services and topics published using the same master on a different computer. '''

    def __init__(self):
        rospy.loginfo('Started agent_relay')

        # Provided services and topics
        service = rospy.Service('agent_relay/instruct', Instruction, self.instruct)
        cancel = rospy.Service('agent_relay/instruct/cancel', Trigger, self.cancel_instruct)
        self.pub = rospy.Publisher('agent_relay/result', InstructionResult, queue_size=200)

        # Used services and topics
        agent_serv = 'agent/instruct'
        rospy.loginfo('AgentRelay waiting for %s service' % agent_serv)
        rospy.wait_for_service(agent_serv)
        self.agent_instruct = rospy.ServiceProxy(agent_serv, Instruction)
        rospy.loginfo('AgentRelay connected to %s' % agent_serv)

        agent_serv = 'agent/instruct/cancel'
        rospy.loginfo('AgentRelay waiting for %s service' % agent_serv)
        rospy.wait_for_service(agent_serv)
        self.agent_cancel = rospy.ServiceProxy(agent_serv, Instruction)
        rospy.loginfo('AgentRelay connected to %s' % agent_serv)

        # Subscribe to the agents trajectory result
        sub = rospy.Subscriber('agent/result', InstructionResult, self.process_result)

        rospy.spin()


    def instruct(self, instr):
        return self.agent_instruct(instr)


    def cancel_instruct(self, req):
        return self.agent_cancel(req)


    def process_result(self, result):
        self.pub.publish(result)


if __name__ == '__main__':

    rospy.init_node('agent_relay', anonymous=False)
    my_node = AgentRelay()


