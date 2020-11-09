#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import rospy
import ros_numpy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from vln_agent.srv import Instruction, InstructionResponse
from vln_agent.msg import InstructionResult

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Twist

from R2R_EnvDrop.model import EncoderLSTM, AttnDecoderLSTM
from R2R_EnvDrop.utils import Tokenizer, read_vocab, quaternion_to_euler, angle_feature




class AgentServer(object):

    def __init__(self):

        rospy.on_shutdown(self.shutdown)

        # Fire up some networks
        self.device = torch.device(rospy.get_param('device', 'cuda:0'))
        self.vocab_path = rospy.get_param('vocab_file')
        self.vocab = read_vocab(self.vocab_path)
        self.max_input = rospy.get_param('max_word_input', 80)
        self.tok = Tokenizer(vocab=self.vocab, encoding_length=self.max_input)
        self.load_model()

        # Services etc
        service = rospy.Service('agent/instruct', Instruction, self.instruct)
        cancel = rospy.Service('agent/instruct/cancel', Trigger, self.cancel_instruct)

        # Subscribe to features and action candidates.
        self.feat_sub = rospy.Subscriber('subgoal/features', Image, self.process_feats)
        self.waypoint_sub = rospy.Subscriber('subgoal/waypoints', PoseArray, self.process_waypoints)
        self.feat_stamp = rospy.Time.now()

        self.instr_id = None # Not executing an instruction
        self.image_paths = [] # Collect the file names of panos seen on the trajectory
        self.image_timestamps = [] # As well as the image timestamps
        self.step = 0
        self.max_steps = 8 # probably expose this as param

        # Connect to theta capture service
        theta_capture = rospy.get_param('pano_capture_service', 'theta/capture')
        rospy.loginfo('Agent waiting for %s service' % theta_capture)
        rospy.wait_for_service(theta_capture)
        self.cam_service = rospy.ServiceProxy(theta_capture, Trigger)
        rospy.loginfo('Agent connected to for %s service' % theta_capture)

        # Connect to move_base
        rospy.loginfo("Agent waiting for move_base")
        self.move_base = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        self.move_base.wait_for_server()
        rospy.loginfo("Agent connected to move_base")

        # Publisher to manually control the robot (e.g. to stop it)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # Publish notifcation when done
        self.pub = rospy.Publisher('agent/result', InstructionResult, queue_size=200)

        rospy.loginfo("Agent ready for instruction")
        rospy.spin()


    def load_model(self):
        ''' Load the PyTorch instruction encoder, action decoder '''
        self.weights_file = rospy.get_param('agent_weights_file')
        model_weights = torch.load(self.weights_file, map_location=self.device)

        def recover_state(name, model):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(model_weights[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(model_weights[name]['state_dict'])
            model.load_state_dict(state)

        self.padding_idx = rospy.get_param('padding_idx', 0)
        wemb = rospy.get_param('word_embedding', 256)
        dropout = rospy.get_param('dropout', 0.5)
        bidir = rospy.get_param('bidirectional', True)
        rnn_dim = rospy.get_param('rnn_dim', 512)
        enc_hidden_size = rnn_dim//2 if bidir else rnn_dim
        self.encoder = EncoderLSTM(self.tok.vocab_size(), wemb, enc_hidden_size, self.padding_idx,
                                          dropout, bidirectional=bidir).to(self.device)
        recover_state('encoder', self.encoder)
        self.encoder.eval()
        rospy.loginfo('Agent loaded encoder')

        aemb = rospy.get_param('action_embedding', 64)
        self.cnn_feat_size = rospy.get_param('cnn_feature_size', 2048)
        self.angle_feat_size = rospy.get_param('angle_feature_size', 128)
        feat_dropout = rospy.get_param('feature_dropout', 0.4)
        self.decoder = AttnDecoderLSTM(aemb, rnn_dim, dropout, feature_size=self.cnn_feat_size+self.angle_feat_size, 
                    angle_feat_size=self.angle_feat_size, feat_dropout=feat_dropout).to(self.device)
        recover_state('decoder', self.decoder)
        self.decoder.eval()
        rospy.loginfo('Agent loaded decoder')


    def process_feats(self, feat_msg):
        ''' Features to torch ready for the network '''
        if self.instr_id is not None:
            self.feat_stamp = feat_msg.header.stamp
            feat = ros_numpy.numpify(feat_msg)
            self.features = torch.from_numpy(feat).to(device=self.device)
            self.features = self.features.transpose(0, 1) # (36+N+1, 2050)
            rospy.loginfo('Agent received features: %s, stamp %.1f' % 
                                          (str(feat.shape), self.feat_stamp.to_sec()))
        else:
            rospy.logwarn('Agent dropped features!')


    def process_waypoints(self, waypoints):
        ''' Build action representation for the network '''
        if self.instr_id is not None:
            self.waypoints = waypoints
            count = 0
            while self.feat_stamp != self.waypoints.header.stamp and count < 30:
                rospy.sleep(0.1)
                count += 1
            if self.feat_stamp == self.waypoints.header.stamp:
                rospy.logdebug('Agent received waypoints, stamp %.1f' % self.feat_stamp.to_sec())
                self.choose_waypoint()
            else:
                rospy.logerr('Agent received waypoints, but no features!')
        else:
            rospy.logwarn('Agent dropped waypoints!')


    def get_input_feat(self):
        ''' Construct inputs to a decoding step of the agent '''
        # What is the agent's current heading? Decoded from first waypoint
        r,p,y = quaternion_to_euler(self.waypoints.poses[-1].orientation)
        agent_matt_heading = 0.5*math.pi-y # In matterport, pos heading turns right from y-axis
        # snap agent heading to 30 degree increments, to match the sim
        headingIncrement = math.pi*2.0/12
        heading_step = int(np.around(agent_matt_heading/headingIncrement))
        if heading_step == 12:
            heading_step = 0
        agent_matt_heading = heading_step * headingIncrement

        # Input action embedding, based only on current heading
        input_a_t = angle_feature(agent_matt_heading, 0, self.angle_feat_size)
        input_a_t = torch.from_numpy(input_a_t).to(device=self.device).unsqueeze(0)

        # Image / candidate feature plus relative orientation encoding in ros coordinates
        feat_matt_heading = 0.5*math.pi-self.features[:,-2]
        feat_elevation = self.features[:,-1]
        feat_rel_heading = feat_matt_heading - agent_matt_heading
        angle_encoding = np.zeros((self.features.shape[0], self.angle_feat_size), dtype=np.float32)
        try:
            for i in range(self.features.shape[0]-1): # Leave zeros in last position (stop vector)
                angle_encoding[i] = angle_feature(feat_rel_heading[i], feat_elevation[i], self.angle_feat_size)
        except:
            import pdb; pdb.set_trace()
        angle_encoding = torch.from_numpy(angle_encoding).to(device=self.device)
        features = torch.cat((self.features[:,:-2], angle_encoding), dim=1).unsqueeze(0)
        f_t = features[:,:36]
        candidate_feat = features[:,36:]

        return input_a_t, f_t, candidate_feat


    def choose_waypoint(self):
        rospy.loginfo('Agent choosing waypoint, step %d' % self.step)

        # We have to stop eventually! This prevents run on.
        if self.step >= self.max_steps:
            self.stop('stop action')

        # Run one step of the decoder network
        input_a_t, f_t, candidate_feat = self.get_input_feat()

        with torch.no_grad():
            self.h_t, self.c_t, logit, self.h1 = self.decoder(input_a_t, f_t, candidate_feat,
                                               self.h_t, self.h1, self.c_t,
                                               self.ctx, self.ctx_mask)
        # Select the best action
        _, a_t = logit.max(1)
        choice = a_t.item()
        probs = F.softmax(logit, 1).squeeze().cpu().numpy()

        if choice == len(self.waypoints.poses)-1:
            # Stop action!
            self.stop('predicted stop action')
        else:
            # call move_base
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose = self.waypoints.poses[choice]
            self.step += 1
            self.move(goal)


    def move(self, goal):
        rospy.logdebug('Agent moving to goal')
        # Send the goal pose to the MoveBaseAction server
        self.move_base.send_goal(goal)
        
        # Allow 1 minute to get there
        finished_within_time = self.move_base.wait_for_result(rospy.Duration(60)) 
        
        # If we don't get there in time, abort the goal
        if not finished_within_time:
            self.move_base.cancel_goal()
            rospy.logwarn("Agent timed out achieving goal, trigger pano anyway")
            #self.stop('move_base timed out')
            self.trigger_camera()
        else:
            # We made it! or we gave up
            state = self.move_base.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.logdebug("Agent reached waypoint")
                self.trigger_camera()
            else:
                rospy.logwarn("Agent failed to reach waypoint, trigger pano anyway")
                #self.stop('move_base failed')
                self.trigger_camera()


    def trigger_camera(self):
        ''' Trigger the pano camera '''
        result = self.cam_service()
        if not result.success:
            err = 'Could not trigger theta camera: %s' % result.message
            rospy.logerr(err)
            return InstructionResponse(success=False, message=err)
        self.image_paths.append(result.message)
        self.image_timestamps.append(rospy.Time.now())
        return InstructionResponse(success=True)


    def stop(self, reason):
        ''' Bookkeeping for stopping an episode '''
        rospy.loginfo('Agent stopping due to: %s' % reason)
        if self.instr_id is not None:
            result = InstructionResult()
            result.header.stamp = rospy.Time.now()
            result.instr_id = self.instr_id
            result.image_filenames = self.image_paths
            result.image_timestamps = self.image_timestamps
            result.reason = reason
            result.start_time = self.start_time
            result.end_time = rospy.Time.now()
            self.pub.publish(result)
            self.instr_id = None
            self.image_paths = []
            self.image_timestamps = []


    def cancel_instruct(self, req):
        ''' Cancel service callback '''
        if self.instr_id:
            #TODO can we cancel move base?
            self.stop('instruction_cancelled')
            return TriggerResponse(success=True)
        else:
            return TriggerResponse(success=False, message='No instruction being processed')


    def instruct(self, req):
        ''' Process a received instruction and trigger the camera '''
        rospy.loginfo('Instr_id %s: %s' % (req.instr_id, req.instruction))
        if self.instr_id is not None:
            err = 'Agent is still processing the last instruction!'
            rospy.logerr(err)
            return InstructionResponse(success=False, message=err)

        self.instr_id = req.instr_id
        self.step = 0
        self.start_time = rospy.Time.now()

        # Trigger the pano camera
        result = self.trigger_camera()
        if not result.success:
            return result

        # While that's happening, process the instruction through the encoder
        self.instruction = req.instruction
        encoding = self.tok.encode_sentence(req.instruction)
        rospy.logdebug('Agent encoded instructed as: %s' % str(self.tok.decode_sentence(encoding)))
        seq_tensor = np.array(encoding).reshape(1,-1)
        seq_lengths = list(np.argmax(seq_tensor == self.padding_idx, axis=1))
        seq_tensor = torch.from_numpy(seq_tensor[:,:seq_lengths[0]]).to(device=self.device).long()
        self.ctx_mask = (seq_tensor == self.padding_idx)[:,:seq_lengths[0]].to(device=self.device).byte()
        with torch.no_grad():
            self.ctx, self.h_t, self.c_t = self.encoder(seq_tensor, seq_lengths)
            self.h1 = self.h_t
        rospy.logdebug('Agent has processed encoder')
        return result


    def shutdown(self):
        # Cancel any active goals
        self.move_base.cancel_goal()
        self.stop('agent was shutdown')
        rospy.sleep(2)
        # Stop the robot
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)



if __name__ == '__main__':

    rospy.init_node('agent', anonymous=False)
    my_node = AgentServer()


