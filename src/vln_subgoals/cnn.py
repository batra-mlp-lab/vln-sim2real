
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as F
import numpy as np
import imp
import math

import rospy
import ros_numpy
from sensor_msgs.msg import Image
from py360convert import e2p  # Old approach used in Caffe CNN pipeline
from Equirec2Perspec import Equirectangular # Faster, used in PyTorch CNN pipeline


class PyTorchCNN(object):
    ''' CNN using standard PyTorch ResNet-152 features from torchvision. ''' 

    def __init__(self):
        ''' Load a pretrained CNN '''
        self.device = torch.device(rospy.get_param('device', 'cuda:0'))
        self.batch_size = rospy.get_param('cnn_batch_size', 1)
        assert 36 % self.batch_size == 0, 'Batch size must evenly divide 36'
        resnet_full = models.resnet152(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet_full.children())[:-1]).to(self.device)
        self.cnn.eval()
        self.pub_views = rospy.Publisher('subgoal/views', Image, queue_size=36)
        rospy.loginfo('CNN loaded')


    def equirectangular_to_perspective(self, image):
        ''' Convert an equirectangular pano (in sensor_msgs.msg.Image format) 
            to 36 perspective images used by the VLN agent. Construct it with 12 heading 
            viewpoints and 3 elevation viewpoints, starting with the heading that is
            directly in front. The agent works better if these views are aligned to the
            building. '''
        equi = Equirectangular(ros_numpy.numpify(image))  # RGB shape of [H, W, 3]
        im_width = rospy.get_param('im_width', 640)
        im_height = rospy.get_param('im_height', 480)
        vfov = rospy.get_param('vfov', 60)

        imgs = torch.zeros((3, 12, 3, im_height, im_width), dtype=torch.float32)
        # heading, elevation. Here, heading is defined from the x-axis, and turning right 
        # is positive. This matches the matterport sim, but opposite to ros
        he = np.zeros((3, 12, 2), dtype=np.float32)

        for i,v_deg in enumerate(range(-30, 60, 30)):
            for j,u_deg in enumerate(range(-90, 270, 30)):
                # Negate heading since for e2p turning right is positive, opposite in ros
                head = -math.radians(u_deg)
                he[i,j] = [head + 2*math.pi if head<0 else head, math.radians(v_deg)]
                p_im = equi.GetPerspective(vfov, u_deg, v_deg, im_height, im_width)
                im = torch.from_numpy(p_im.astype(np.float32).transpose((2, 0, 1)))/255.0
                imgs[i,j] = F.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # RGB normalization
        return imgs.reshape(-1, 3, im_height, im_width), he.reshape(-1, 2)


    def extract_features(self, image, stamp=None, augment=True):
        ''' Extract cnn features from a ros Image object, and also return heading and
            elevation in radians relative to the centre of the image (which is ros coordinates,
            center of the image is on the x-axis). Output shape is [feat_dim, 36], [2,36]'''
        im_data, imgs_he = self.equirectangular_to_perspective(image)
        im_data = im_data.to(device=self.device)

        feat_dim = rospy.get_param('cnn_feature_dim', 2048)
        feats = torch.empty((feat_dim, im_data.shape[0]), dtype=torch.float32).to(device=self.device)
        with torch.no_grad():
            for n in range(0,im_data.shape[0],self.batch_size):
                feats[:,n:n+self.batch_size] = self.cnn(im_data[n:n+self.batch_size]).reshape(self.batch_size,-1).transpose(0,1)
        return feats, imgs_he.transpose()


class CaffeCNN(object):
    ''' CNN using Caffe ResNet-152 features (matching the pre-extracted CNN
        features provided with the Matterport3DSimulator). In this version the
        caffe model weights have been converted to PyTorch.''' 

    def __init__(self):
        ''' Load a pretrained CNN '''
        self.device = torch.device(rospy.get_param('device', 'cuda:0'))
        self.cnn_weights = rospy.get_param('cnn_weights_file')
        self.cnn_arch = rospy.get_param('cnn_arch_file')
        self.batch_size = rospy.get_param('cnn_batch_size', 1)
        assert 36 % self.batch_size == 0, 'Batch size must evenly divide 36'
        MainModel = imp.load_source('MainModel', self.cnn_arch)
        self.cnn = torch.load(self.cnn_weights).to(self.device)
        self.cnn.eval()
        self.pub_views = rospy.Publisher('subgoal/views', Image, queue_size=36)
        rospy.loginfo('CNN loaded')


    def equirectangular_to_perspective(self, image):
        ''' Convert an equirectangular pano (in sensor_msgs.msg.Image format) 
            to 36 perspective images used by the VLN agent. Construct it with 12 heading 
            viewpoints and 3 elevation viewpoints, starting with the heading that is
            directly in front. The agent works better if these views are aligned to the
            building. '''
        image = ros_numpy.numpify(image)
        im_width = rospy.get_param('im_width', 640)
        im_height = rospy.get_param('im_height', 480)
        vfov = rospy.get_param('vfov', 60)
        hfov = math.degrees(2*math.atan(im_width/im_height * math.tan(0.5*math.radians(vfov))))
        fov_deg = (hfov, vfov)
        out_hw = (im_height, im_width)

        imgs = np.zeros((3, 12, im_height, im_width, 3), dtype=np.float32)
        # heading, elevation. Here, heading is defined from the x-axis, and turning right 
        # is positive. This matches the matterport sim, but opposite to ros
        he = np.zeros((3, 12, 2), dtype=np.float32)

        for i,v_deg in enumerate(range(-30, 60, 30)):
            for j,u_deg in enumerate(range(-90, 270, 30)):
                # Negate heading since for e2p turning right is positive, opposite in ros
                head = -math.radians(u_deg)
                he[i,j] = [head + 2*math.pi if head<0 else head, math.radians(v_deg)]
                imgs[i,j] = e2p(image, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear')
        return imgs.reshape(-1, im_height, im_width, 3), he.reshape(-1, 2)


    def extract_features(self, image, stamp=None, augment=True):
        ''' Extract cnn features from a ros Image object, and also return heading and
            elevation in radians relative to the centre of the image (which is ros coordinates,
            center of the image is on the x-axis). Output shape is [feat_dim, 36], [2,36]'''
        ims, imgs_he = self.equirectangular_to_perspective(image)
        if rospy.get_param('publish_views', False): # more expensive than the others
            if stamp == None:
                stamp = rospy.Time.now()
            for i in range(ims.shape[0]):
                view = ros_numpy.msgify(Image, ims[i].astype(np.uint8), encoding=image.encoding)
                view.header.stamp = stamp
                self.pub_views.publish(view)

        im_data = ims.transpose(0,3,1,2)
        im_data -= np.array([123.2, 115.9, 103.1]).reshape(1,3,1,1) # RGB pixel mean - #TODO move to parameter
        im_data = torch.from_numpy(im_data).to(device=self.device).flip([1]) # flip RGB to BGR for net

        feat_dim = rospy.get_param('cnn_feature_dim', 2048)
        feats = torch.empty((feat_dim, im_data.shape[0]), dtype=torch.float32).to(device=self.device)
        with torch.no_grad():
            for n in range(0,im_data.shape[0],self.batch_size):
                feats[:,n:n+self.batch_size] = self.cnn(im_data[n:n+self.batch_size]).reshape(self.batch_size,-1).transpose(0,1)
        return feats, imgs_he.transpose()


if __name__ == "__main__":

    # Crude test of PyTorchCNN. Make sure to run 'roscore'.
    py_cnn = PyTorchCNN()
    caffe_cnn = CaffeCNN()

    import cv2
    from cv_bridge import CvBridge
    img = cv2.imread('sample_theta_images/f8a8df104bd34f77b9c58798d4cc9440_equirectangular.jpg')
    bridge = CvBridge()
    imgMsg = bridge.cv2_to_imgmsg(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "rgb8")
    
    feats, imgs_he = py_cnn.extract_features(imgMsg)
    caffe_cnn.equirectangular_to_perspective(imgMsg)


