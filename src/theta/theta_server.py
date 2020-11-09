#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
import numpy as np
import gphoto2 as gp
import io
from PIL import Image as Img
from collections import OrderedDict


import os
import sys
from subprocess import Popen, PIPE
import fcntl

USBDEVFS_RESET= 21780


def reset_usb_driver():
    ''' Try to overcome issues with the Theta camera '''
    rospy.loginfo('Resetting Ricoh usb device')
    try:
        lsusb_out = Popen("lsusb | grep -i Ricoh", shell=True, bufsize=64, stdin=PIPE, stdout=PIPE, close_fds=True).stdout.read().strip().split()
        bus = lsusb_out[1]
        device = lsusb_out[3][:-1]
        f = open("/dev/bus/usb/%s/%s"%(bus, device), 'w', os.O_WRONLY)
        fcntl.ioctl(f, USBDEVFS_RESET, 0)
        rospy.loginfo('Reset bus %s device %s' % (bus,device))
    except Exception, msg:
        rospy.logwarn('failed to reset device: %s' % msg)


# Helper functions from https://github.com/jim-easterbrook/python-gphoto2/blob/master/examples/cam-conf-view-gui.py

def get_camera_model(camera_config):
    # get the camera model
    OK, camera_model = gp.gp_widget_get_child_by_name(
        camera_config, 'cameramodel')
    if OK < gp.GP_OK:
        OK, camera_model = gp.gp_widget_get_child_by_name(
            camera_config, 'model')
    if OK >= gp.GP_OK:
        camera_model = camera_model.get_value()
    else:
        camera_model = ''
    return camera_model


def get_camera_config_children(childrenarr, savearr, propcount):
    for child in childrenarr:
        tmpdict = OrderedDict()
        propcount.numtot += 1
        if child.get_readonly():
            propcount.numro += 1
        else:
            propcount.numrw += 1
        tmpdict['idx'] = str(propcount)
        tmpdict['ro'] = child.get_readonly()
        tmpdict['name'] = child.get_name()
        tmpdict['label'] = child.get_label()
        tmpdict['type'] = child.get_type()
        #tmpdict['typestr'] = get_gphoto2_CameraWidgetType_string( tmpdict['type'] )
        if ((tmpdict['type'] == gp.GP_WIDGET_RADIO) or (tmpdict['type'] == gp.GP_WIDGET_MENU)):
            tmpdict['count_choices'] = child.count_choices()
            tmpchoices = []
            for choice in child.get_choices():
                tmpchoices.append(choice)
            tmpdict['choices'] = ",".join(tmpchoices)
        if (child.count_children() > 0):
            tmpdict['children'] = []
            get_camera_config_children(child.get_children(), tmpdict['children'], propcount)
        else:
            # NOTE: camera HAS to be "into preview mode to raise mirror", otherwise at this point can get "gphoto2.GPhoto2Error: [-2] Bad parameters" for get_value
            try:
                tmpdict['value'] = child.get_value()
            except Exception as ex:
                tmpdict['value'] = "{} {}".format( type(ex).__name__, ex.args)
                propcount.numexc += 1
        savearr.append(tmpdict)

def get_camera_config_object(camera_config):
    retdict = OrderedDict()
    retdict['camera_model'] = get_camera_model(camera_config)
    propcount = PropCount()
    retarray = []
    retdict['children'] = []
    get_camera_config_children(camera_config.get_children(), retdict['children'], propcount)
    excstr = "no errors - OK." if (propcount.numexc == 0) else "{} errors - bad (please check if camera mirror is up)!".format(propcount.numexc)
    print("Parsed camera config: {} properties total, of which {} read-write and {} read-only; with {}".format(propcount.numtot, propcount.numrw, propcount.numro, excstr))
    return retdict


class PropCount(object):
    def __init__(self):
        self.numro = 0
        self.numrw = 0
        self.numtot = 0
        self.numexc = 0
    def __str__(self):
        return "{},{},{},{}".format(self.numtot,self.numrw,self.numro,self.numexc)


class ThetaServer(object):


    def init_camera(self):
        # Fire up the camera
        self.context = gp.Context()
        self.camera = gp.Camera()

        while not rospy.is_shutdown():
            error = gp.gp_camera_init(self.camera, self.context)
            if error >= gp.GP_OK:
                # operation completed successfully so exit loop
                break
            if error != gp.GP_ERROR_MODEL_NOT_FOUND:
                # some other error we can't handle here
                raise gp.GPhoto2Error(error)
            # no camera, try again in 2 seconds
            rospy.logwarn('Please connect and switch on the Ricoh Theta camera')
            rospy.sleep(2)

        summary = self.camera.get_summary()
        rospy.loginfo('Detected camera: %s, %s' % tuple(str(summary).split('\n')[:2]))
        self.camera_config = None

        while not self.camera_config and not rospy.is_shutdown():
            try:
                self.camera_config = self.camera.get_config()
            except:
                rospy.logwarn('Please switch on the Ricoh Theta camera')
                rospy.sleep(2)
                continue
            OK, battery_level = gp.gp_widget_get_child_by_name(self.camera_config, 'batterylevel')
            if OK >= gp.GP_OK:
                rospy.loginfo('Camera Battery level: %s' % battery_level.get_value())
            else:
                rospy.loginfo('Could not access Camera Battery level')

    def __init__(self):

        self.init_camera()

        # Service
        self.service = rospy.Service('theta/capture', Trigger, self.capture)

        # Publishers
        self.pub_image = rospy.Publisher('theta/image', Image, queue_size=1)

        tilt_sensor = rospy.get_param('theta_tilt_sensor', True)
        tilt_window = rospy.get_param('theta_tilt_window', 10) # around half a second
        self.tilt_thresh = rospy.get_param('theta_tilt_threshold', 0.00001)
        self.imu_buffer = np.zeros((3,tilt_window), dtype=np.float32)
        if tilt_sensor:
            self.imu = rospy.Subscriber('/mobile_base/sensors/imu_data', Imu, self.process_imu)

        rospy.spin()


    def process_imu(self, data):
        ''' Keep a sensor to see if the base is rocking or not '''
        self.imu_buffer = np.roll(self.imu_buffer,1)
        self.imu_buffer[0,0] = data.angular_velocity.x
        self.imu_buffer[1,0] = data.angular_velocity.y
        self.imu_buffer[2,0] = data.angular_velocity.z


    def rocking(self):
        return np.any(np.max(np.abs(self.imu_buffer), axis=1) > self.tilt_thresh)


    def single_capture(self):
        count = 0
        while self.rocking() and count < 10:
            rospy.sleep(0.5)
            rospy.logwarn('Theta waiting for rocking to stop')
            count += 1
        try:
            self.file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
            self.stamp = rospy.Time.now()
            self.camera_file = self.camera.file_get(self.file_path.folder, self.file_path.name, gp.GP_FILE_TYPE_NORMAL)
            return True
        except Exception, e:
            rospy.logerr(str(e))
            return False


    def capture(self, req):
        rospy.loginfo('Capturing panorama')
        if not self.single_capture():
            rospy.logerr('Could not trigger Richo Theta, retrying in 1s')
            rospy.sleep(1)
            while not self.single_capture():
                  rospy.logerr('Could not trigger Richo Theta, resetting the usb driver. Is the camera switched on and set to still image capture mode?')
                  reset_usb_driver()
                  self.init_camera()
                  rospy.sleep(2)

        # Construct image
        file_data = gp.check_result(gp.gp_file_get_data_and_size(self.camera_file))
        img = Img.open(io.BytesIO(file_data))
        rospy.loginfo('Panorama captured!')

        image = Image(height=img.height, width=img.width, encoding="rgb8", is_bigendian=False, step=img.width*3, data=img.tobytes())
        image.header.stamp = self.stamp
        image.header.frame_id = 'map' #TODO maybe add something sensible here

        self.pub_image.publish(image)
        rospy.loginfo('Panorama published!')

        # Avoid running out of space. Update: No need: Device holds approx 4,800 photos
        #try:
        #    self.camera.file_delete(self.file_path.folder, self.file_path.name)
        #except:
        #    rospy.logwarn('Delete photo on the Ricoh Theta failed. Camera may eventually run out of storage.')
        return TriggerResponse(success=True, message=self.file_path.folder+'/'+self.file_path.name)




if __name__ == '__main__':

    rospy.init_node('theta', anonymous=False)
    my_node = ThetaServer()


