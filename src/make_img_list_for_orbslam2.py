#!/home/user/anaconda3/envs/ros/bin/python
#-*- encoding: utf8 -*-
import os
import sys
sys.path.insert(0, os.getcwd())

import functools
import rospy
import tf2_ros
import torch
import cv2
import time

from std_msgs.msg import Empty, String, Bool, Header, Float64
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation

from DenseFusion._init_paths import *
from DenseFusion.eval_ycb import *
# Scene divide

class Densefusion:
    def __init__(self):
        
        # self.wait_scene_done = rospy.Subscriber('wait_done_scene', String, self.handle_scene)
        self.result_pub = rospy.Publisher("/densefusion_result", Image, queue_size=1)
        self.image_raw_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
        self.image_test_pub = rospy.Publisher("/camera/color/image_test", Image, queue_size=10)
        self.depth_raw_pub = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)
        self.depth_test_pub = rospy.Publisher("/camera/depth/image_test", Image, queue_size=10)
        self.point_cloud_pub = rospy.Publisher("/point_cloud_densefusion", PointCloud2, queue_size=1)
        self.world_point_cloud_pub = rospy.Publisher("/world_point_cloud_densefusion", PointCloud2, queue_size=1)
        self.interesting_points_pub = rospy.Publisher("/interesting_points_densefusion", PointCloud2, queue_size=1)

        self.br = CvBridge()


    def handle_image_sub(self, msg):
        # self.image = self.br.imgmsg_to_cv2(msg)
        # print("image shape : ", self.image.shape)
        return
    def handle_depth_sub(self, msg):
        # self.depth = self.br.imgmsg_to_cv2(msg)
        # print("depth shape : ", self.depth.shape)
        return

    def make_img_list(self):
        save_txt = ""
        for now in range(2088, 2375):
            save_txt += str(rospy.Time.now().to_sec()) + " " + '{0}/{1}-color.png'.format(dataset_root_dir, testlist[now])\
            + " " + str(rospy.Time.now().to_sec()) + " " + '{0}/{1}-depth.png'.format(dataset_root_dir, testlist[now]) + '\n'

            camera_meta = scio.loadmat('{0}/{1}-meta.mat'.format(dataset_root_dir, testlist[now]))
            extrinsic = np.array(camera_meta['rotation_translation_matrix'])
            E_R = extrinsic[:,:3]
            E_T = extrinsic[:,3]
            E_T = E_T.reshape(3,1)
            external_row = np.matrix([0, 0, 0, 1])
            TransE = np.hstack((E_R, E_T))
            TransE = np.vstack((TransE, external_row))


            if now == 2088:
                print TransE
                print inv(TransE)
                print('=================================')
            if now == 2374:
                print TransE
                print inv(TransE)
            # time.sleep(0.03)
        
        # save_file = open('/home/user/packages/ORB_SLAM2/Examples/RGB-D/associations/YCB.txt', mode='wt')
        # save_file.write(save_txt)
        # save_file.close()


if __name__ == '__main__':
    
    # bc = np.load('/home/user/catkin_ws/src/densefusion_py/interesting_points/scene10/bleach_cleanser.npy')
    # print bc[0]
    # print bc.shape

    rospy.init_node('py_densefusion', anonymous=False)
    df = Densefusion()
    # df.densefusion()
    # df.densefusion_live()
    df.make_img_list()

    #py_trees.logging.level = py_trees.logging.Level.DEBUG

    # print("starting..")
    # print('initialize...')

    # rospy.spin()
