#!/home/user/anaconda3/envs/ros/bin/python
#-*- encoding: utf8 -*-
import os
import sys
sys.path.insert(0, os.getcwd())

import functools
import rospy
import tf2_ros
import cv2
import time

# from yolact.data.coco import COCODetection, get_label_map, MEANS, COLORS, cfg, set_cfg, set_dataset

from std_msgs.msg import Empty, String, Bool, Header, Float64
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation

# Scene divide


class Densefusion:
    def __init__(self):
        
        # self.wait_scene_done = rospy.Subscriber('wait_done_scene', String, self.handle_scene)
        self.result_pub = rospy.Publisher("/densefusion_result", Image, queue_size=1)
        self.image_raw_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
        self.image_test_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=10)
        self.depth_raw_pub = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)
        self.depth_test_pub = rospy.Publisher("/camera/depth_registered/image_raw", Image, queue_size=10)
        self.point_cloud_pub = rospy.Publisher("/point_cloud_densefusion", PointCloud2, queue_size=1)
        self.world_point_cloud_pub = rospy.Publisher("/world_point_cloud_densefusion", PointCloud2, queue_size=1)
        self.interesting_points_pub = rospy.Publisher("/interesting_points_densefusion", PointCloud2, queue_size=1)

        self.image_raw_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.handle_image_sub)
        self.depth_raw_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.handle_depth_sub)
        self.br = CvBridge()

        self.mImage = None
        self.mDepth = None

        self.bPubImgLock = False
        self.iCurrentSeq = 0

        self.association_file_path = '/home/user/packages/ORB_SLAM2/Examples/RGB-D/associations/fr1_xyz.txt'
        self.association_file = open(self.association_file_path, mode='rt')
        self.splitlinestr = str.splitlines(self.association_file.read())
        self.image_path = '/home/user/packages/ORB_SLAM2/Examples/Monocular/rgbd_dataset_freiburg1_xyz/'

        self.association_file.close()



    def handle_image_sub(self, msg):
        return
    def handle_depth_sub(self, msg):
        return

    def image_pub(self):
        for str_line in self.splitlinestr:
            image_path = str_line
            time_stamp_color = image_path.split()[0]
            image_path_color = self.image_path + image_path.split()[1]
            time_stamp_depth = image_path.split()[2]
            image_path_depth = self.image_path + image_path.split()[3]

            # print image_path_color
            # print image_path_depth

            img = cv2.imread(image_path_color, -1)
            depth = cv2.imread(image_path_depth, -1)

            # print depth
            # exit()

            img_msg = self.br.cv2_to_imgmsg(img)
            img_msg.header.frame_id = "camera_rgb_optical_frame"
            # img_msg.header.stamp = rospy.Time.now()
            t = rospy.Time.from_sec(float(time_stamp_color))
            img_msg.header.stamp = t

            depth_msg = self.br.cv2_to_imgmsg(depth)
            depth_msg.header.frame_id = "camera_depth_optical_frame"
            # depth_msg.header.stamp = rospy.Time.now()
            t = rospy.Time.from_sec(float(time_stamp_depth))
            depth_msg.header.stamp = t

            # print time_stamp_color.toSec()
            # print img_msg.header.stamp

            self.image_test_pub.publish(img_msg)
            self.depth_test_pub.publish(depth_msg)

            time.sleep(0.1)


if __name__ == '__main__':
    rospy.init_node('py_densefusion', anonymous=False)
    df = Densefusion()
    df.image_pub()

