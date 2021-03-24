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

# from yolact.data.coco import COCODetection, get_label_map, MEANS, COLORS, cfg, set_cfg, set_dataset

from DenseFusion._init_paths import *
from DenseFusion.eval_ycb import *

from std_msgs.msg import Empty, String, Bool, Header, Float64
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation

# Scene divide


def make_new_pointcloud2_msg(pointcloud_msg, points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    if stamp:
        pointcloud_msg.header.stamp = stamp
    if frame_id:
        pointcloud_msg.header.frame_id = frame_id
    if seq:
        pointcloud_msg.header.seq = seq
    if len(points.shape) == 3:
        pointcloud_msg.height = points.shape[1]
        pointcloud_msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        pointcloud_msg.height = 1
        pointcloud_msg.width = N

    pointcloud_msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    pointcloud_msg.is_bigendian = False
    pointcloud_msg.point_step = 24
    pointcloud_msg.row_step = pointcloud_msg.point_step * N
    pointcloud_msg.is_dense = True
    pointcloud_msg.data = xyzrgb.tostring()

    return pointcloud_msg 

def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = xyzrgb.tostring()

    return msg 

class Densefusion:
    def __init__(self):
        
        # self.wait_scene_done = rospy.Subscriber('wait_done_scene', String, self.handle_scene)
        self.result_pub = rospy.Publisher("/densefusion_result", Image, queue_size=1)
        self.image_raw_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
        # self.image_test_pub = rospy.Publisher("/camera/color/image_test", Image, queue_size=10)
        self.image_test_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=10)
        self.depth_raw_pub = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)
        # self.depth_test_pub = rospy.Publisher("/camera/depth/image_test", Image, queue_size=10)
        self.depth_test_pub = rospy.Publisher("/camera/depth_registered/image_raw", Image, queue_size=10)
        self.point_cloud_pub = rospy.Publisher("/point_cloud_densefusion", PointCloud2, queue_size=1)
        self.world_point_cloud_pub = rospy.Publisher("/world_point_cloud_densefusion", PointCloud2, queue_size=1)
        self.interesting_points_pub = rospy.Publisher("/interesting_points_densefusion", PointCloud2, queue_size=1)

        self.image_raw_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.handle_image_sub)
        self.depth_raw_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.handle_depth_sub)
        # self.camera_pose_sub = rospy.Subscriber("/orb_slam2_rgbd/pose", PoseStamped, self.handle_pose_sub)
        # self.camera_pose_sub = rospy.Subscriber("/orb_slam2_rgbd/str_type_pose", String, self.handle_pose_sub)
        self.camera_pose_sub = rospy.Subscriber("/orb_slam3/str_type_pose", String, self.handle_pose_sub)
        self.camera_pose_pub = rospy.Publisher("/YCB_cam_pose", PoseStamped, queue_size=1)
        self.orb_camera_pose_pub = rospy.Publisher("/orb_cam_pose", PoseStamped, queue_size=1)
        self.br = CvBridge()

        self.mImage = None
        self.mDepth = None

        self.bPubImgLock = False
        self.iCurrentSeq = 0


        self.is_first = True
        self.first_world_model = None
        self.first_world_model_color = None

        self.camera_model = draw_cld[11]
        self.cam_external_dim = np.ones_like(self.camera_model[:,0]).reshape(self.camera_model[:,0].size, 1)
        self.camera_model = np.hstack([self.camera_model, self.cam_external_dim])

        self.interesting_points = {}
        self.interesting_colors = {}
        for i in range(22):
            self.interesting_points[i] = None
            self.interesting_colors[i] = None

        self.prev_E_T = None
        self.prev_E_R = None

        self.start = False

        self.E_R = np.zeros((3,3))
        self.E_T = np.zeros((3,))
        self.TransE = np.eye(4)
        # self.densefusion

    def handle_image_sub(self, msg):
        # self.image = self.br.imgmsg_to_cv2(msg)
        # print("image shape : ", self.image.shape)
        return
    def handle_depth_sub(self, msg):
        # self.depth = self.br.imgmsg_to_cv2(msg)
        # print("depth shape : ", self.depth.shape)
        return

    def image_pub(self):
        pub_rate = rospy.Rate(30)
        # while(not rospy.is_shutdown()):
        # for now in range(394):   # Scene 1
        # for now in range(395, 621):   # Scene 2
        # for now in range(622, 811):   # Scene 3
        for now in range(2088, 2375): # Scene 10
        # for now in range(1899, 2087):   # Scene 9
        # for now in range(2100, 2151):
            # while(self.bPubImgLock):
            #     time.sleep(0.01)
            #     # print "waiting..."
            # img = PIL_Image.open('{0}/{1}-color.png'.format(dataset_root_dir, testlist[now]))
            # depth = np.array(PIL_Image.open('{0}/{1}-depth.png'.format(dataset_root_dir, testlist[now])))

            # # print depth

            # pub_img = img.copy()
            # pub_img = np.array(pub_img)[:, :, :3]
            # pub_img.astype(np.float32)

            # pub_depth = depth.copy()
            # pub_depth = np.array(pub_depth)
            # pub_depth = pub_depth.astype(np.float)
            # pub_depth *= cam_scale
            # print pub_depth
            # print now

            img = cv2.imread('{0}/{1}-color.png'.format(dataset_root_dir, testlist[now]), -1)
            depth = cv2.imread('{0}/{1}-depth.png'.format(dataset_root_dir, testlist[now]), -1)

            # print depth.shape
            # exit()

            img_msg = self.br.cv2_to_imgmsg(img)
            img_msg.header.frame_id = "camera_rgb_optical_frame"
            img_msg.header.stamp = rospy.Time.now()

            depth_msg = self.br.cv2_to_imgmsg(depth)
            depth_msg.header.frame_id = "camera_depth_optical_frame"
            depth_msg.header.stamp = rospy.Time.now()

            self.image_test_pub.publish(img_msg)
            self.depth_test_pub.publish(depth_msg)
            self.iCurrentSeq = now

            # print img_msg.header.stamp
            # print depth_msg.header.stamp

            self.mImage = PIL_Image.open('{0}/{1}-color.png'.format(dataset_root_dir, testlist[now]))
            self.mDepth = np.array(PIL_Image.open('{0}/{1}-depth.png'.format(dataset_root_dir, testlist[now])))
            time.sleep(0.1)


            camera_meta = scio.loadmat('{0}/{1}-meta.mat'.format(dataset_root_dir, testlist[now]))
            extrinsic = np.array(camera_meta['rotation_translation_matrix'])
            E_R = extrinsic[:,:3]
            E_T = extrinsic[:,3]
            # E_T = E_T.reshape(3,1)

            r = Rotation.from_dcm(E_R)
            quat = r.as_quat()
            # print quat
            camera_pose = PoseStamped()

            camera_pose.header.stamp = rospy.Time.now()
            camera_pose.header.frame_id = "map"

            camera_pose.pose.position.x = E_T[0]
            camera_pose.pose.position.y = E_T[1]
            camera_pose.pose.position.z = E_T[2]

            camera_pose.pose.orientation.x = quat[0]
            camera_pose.pose.orientation.y = quat[1]
            camera_pose.pose.orientation.z = quat[2]
            camera_pose.pose.orientation.w = quat[3]

            self.camera_pose_pub.publish(camera_pose)

        # self.start = True
        # for now in range(2200, 2088, -1):
        #     while(self.bPubImgLock):
        #         time.sleep(0.01)
        #     img = PIL_Image.open('{0}/{1}-color.png'.format(dataset_root_dir, testlist[now]))
        #     depth = np.array(PIL_Image.open('{0}/{1}-depth.png'.format(dataset_root_dir, testlist[now])))

        #     # print depth

        #     pub_img = img.copy()
        #     pub_img = np.array(pub_img)[:, :, :3]
        #     pub_img.astype(np.float32)

        #     pub_depth = depth.copy()
        #     pub_depth = np.array(pub_depth)
        #     pub_depth = pub_depth.astype(np.float)
        #     pub_depth *= cam_scale
        #     # print now
        #     img_msg = self.br.cv2_to_imgmsg(pub_img)
        #     img_msg.header.frame_id = "camera_rgb_optical_frame"
        #     img_msg.header.stamp = rospy.Time.now()

        #     depth_msg = self.br.cv2_to_imgmsg(pub_depth)
        #     depth_msg.header.frame_id = "camera_depth_optical_frame"
        #     depth_msg.header.stamp = rospy.Time.now()

        #     self.image_test_pub.publish(img_msg)
        #     self.depth_test_pub.publish(depth_msg)
        #     self.iCurrentSeq = now

        #     self.mImage = img
        #     self.mDepth = depth
        #     time.sleep(0.1)


    def handle_pose_sub(self, msg):
        now = self.iCurrentSeq
        print "now: " + str(now) + " seq"
        # if now < 2110:
        #     return
        # if not self.start:
        #     return
        # if now >= 2110:
        #     print msg
        #     return
        # if now == 2120:
        #     print msg
        #     exit()
        self.bPubImgLock = True

        # print msg

            
        img = self.mImage
        depth = self.mDepth
        posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
        label = np.array(posecnn_meta['labels'])
        posecnn_rois = np.array(posecnn_meta['rois'])
        camera_meta = scio.loadmat('{0}/{1}-meta.mat'.format(dataset_root_dir, testlist[now]))
        extrinsic = np.array(camera_meta['rotation_translation_matrix'])


        # E_R = quaternion_matrix(np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]))
        # E_T = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        TransE = np.zeros((4,4))
        TransE[0,0] = float(msg.data.split()[0])
        TransE[0,1] = float(msg.data.split()[1])
        TransE[0,2] = float(msg.data.split()[2])

        TransE[1,0] = float(msg.data.split()[3])
        TransE[1,1] = float(msg.data.split()[4])
        TransE[1,2] = float(msg.data.split()[5])

        TransE[2,0] = float(msg.data.split()[6])
        TransE[2,1] = float(msg.data.split()[7])
        TransE[2,2] = float(msg.data.split()[8])

        TransE[0,3] = float(msg.data.split()[9])
        TransE[1,3] = float(msg.data.split()[10])
        TransE[2,3] = float(msg.data.split()[11])

        TransE[3,3] = 1.

        # print msg.data
        # print type(msg.data)
        # print msg.data.split()

        # E_T = np.array([msg.pose.position.z, -msg.pose.position.x, -msg.pose.position.y])
        # E_T = extrinsic[:,3]
        # TransE[:3,3] = E_T
        
        # print TransE

        # TransE = np.zeros((4,4))
        # TransE[0,0] = mTransE[1,1]
        # TransE[0,1] = mTransE[2,1]
        # TransE[0,2] = -mTransE[0,1]

        
        # TransE[1,0] = mTransE[1,2]
        # TransE[1,1] = mTransE[2,2]
        # TransE[1,2] = -mTransE[0,2]

        # TransE[2,0] = -mTransE[1,0]
        # TransE[2,1] = -mTransE[2,0]
        # TransE[2,2] = mTransE[0,0]

        # TransE[:3,3] = E_T
        # TransE[3,3] = 1



        # E_T = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # self.E_T += np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # E_R = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # E_R = quaternion_matrix(np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]))
        # E_R = quaternion_matrix(np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]))

        # self.E_R += quaternion_matrix(np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]))[:3,:3]

        # E_T = self.E_T.reshape(3,1)
        # external_row = np.matrix([0, 0, 0, 1])
        # TransE = np.hstack((self.E_R, E_T))
        # TransE = np.vstack((TransE, external_row))

        # TransE = E_R
        # # TransE = inv(TransE)
        # E_R = extrinsic[:,:3]
        # TransE[:3,:3] = E_R
        # E_T = extrinsic[:,3]
        # # E_T = E_T.reshape(3,1)
        # TransE[:3,3] = E_T

        # print E_R

                # external_row = np.matrix([0, 0, 0, 1])
                # TransE = np.hstack((E_R, E_T))
                # TransE = np.vstack((TransE, external_row))

        r = Rotation.from_dcm(inv(TransE[:3,:3]))
        quat = r.as_quat()
        # print quat
        camera_pose = PoseStamped()

        camera_pose.header.stamp = rospy.Time.now()
        camera_pose.header.frame_id = "map"

        camera_pose.pose.position.x = inv(TransE[3,0]
        camera_pose.pose.position.y = inv(TransE[3,1]
        camera_pose.pose.position.z = inv(TransE[3,2]
        # camera_pose.pose.position.x = float(msg.data.split()[9])
        # camera_pose.pose.position.y = float(msg.data.split()[10])
        # camera_pose.pose.position.z = float(msg.data.split()[11])
        # camera_pose.pose.position.x = E_T[0]
        # camera_pose.pose.position.y = E_T[1]
        # camera_pose.pose.position.z = E_T[2]

        camera_pose.pose.orientation.x = quat[0]
        camera_pose.pose.orientation.y = quat[1]
        camera_pose.pose.orientation.z = quat[2]
        camera_pose.pose.orientation.w = quat[3]

        self.orb_camera_pose_pub.publish(camera_pose)

        # print quaternion_matrix(quat)

        # TransE[0,3] = E_T[2]
        # TransE[1,3] = -E_T[0]
        # TransE[2,3] = -E_T[1]
        # print TransE
        # r = Rotation.from_dcm(TransE[:3,:3])
        # r = Rotation.from_quat(np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]))
        # print r.as_dcm()
        # print "Euler xyz"
        # print r.as_euler('xyz', degrees=True)

        # self.bPubImgLock = False
        # return

        # TransE = inv(TransE)

        pred_points = None
        pred_colors = None
 
        world_points = None
        world_colors = None
        
        external_row = np.matrix([0, 0, 0, 1])

        lst = posecnn_rois[:, 1:2].flatten()
        reverse_mask = ma.getmaskarray(ma.masked_equal(label, 0))

        draw_img = img.copy()
        draw_img = np.array(draw_img)[:, :, :3]
        draw_img.astype(np.float32)
        matching_score_zeros_map = np.zeros((img_height, img_width, 3))
        
        # print TransE
        for idx in range(len(lst)):
            itemid = int(lst[idx])
            try:
                print " *** " , class_name[itemid -1] , " ***"
                rmin, rmax, cmin, cmax = get_bbox(posecnn_rois, idx)

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
                mask = mask_label * mask_depth

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])

                pt2 = depth_masked * cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                img_masked = np.array(img)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([itemid - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                cnn_img, pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

                pred_c = pred_c.view(bs, num_points)
                how_max, which_max = torch.max(pred_c, 1)
                pred_t = pred_t.view(bs * num_points, 1, 3)
                points = cloud.view(bs * num_points, 1, 3)

                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)

                for ite in range(0, iteration):
                    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                    my_mat = quaternion_matrix(my_r)
                    R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                    my_mat[0:3, 3] = my_t
                    
                    new_cloud = torch.bmm((cloud - T), R).contiguous()
                    pred_r, pred_t = refiner(new_cloud, emb, index)
                    pred_r = pred_r.view(1, 1, -1)
                    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                    my_r_2 = pred_r.view(-1).cpu().data.numpy()
                    my_t_2 = pred_t.view(-1).cpu().data.numpy()
                    my_mat_2 = quaternion_matrix(my_r_2)

                    my_mat_2[0:3, 3] = my_t_2

                    my_mat_final = np.dot(my_mat, my_mat_2)
                    my_r_final = copy.deepcopy(my_mat_final)
                    my_r_final[0:3, 3] = 0
                    my_r_final = quaternion_from_matrix(my_r_final, True)
                    my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                    my_pred = np.append(my_r_final, my_t_final)
                    my_r = my_r_final
                    my_t = my_t_final


                draw_img = draw_object(int(itemid), my_r, my_t, draw_img)

                # TransM : camera to model transform matrix
                M_R = quaternion_matrix(my_r)[:3, :3]
                M_T = my_t
                M_T = M_T.reshape(3,1)
                TransM = np.hstack((M_R, M_T))
                TransM = np.vstack((TransM, external_row))

                model_points = cld[itemid]
                external_dim = np.ones_like(model_points[:,0]).reshape(model_points[:,0].size, 1)
                model_points = np.hstack([model_points, external_dim])

                pred = np.dot(TransM, model_points.T).T         # Model points in camera coordinate
                world_model = np.dot(inv(TransE), pred.T).T     # Model points in world coordinate
                # print pred.shape
                # world_model = np.dot(pred[:,:3], E_R) + E_T

                # Check camera trajectory using projection.
                if self.is_first:
                    if self.first_world_model is None:
                        self.first_world_model = world_model
                        self.first_world_model_color = np.tile(color_norm[int(itemid-1)], (world_model.shape[0],1))
                    else:
                        self.first_world_model = np.vstack((self.first_world_model, world_model))
                        self.first_world_model_color = np.vstack((self.first_world_model_color, np.tile(color_norm[int(itemid-1)], (world_model.shape[0],1))))
                    # print self.first_world_model.shape
                    # self.is_first = False

                # if pred_points is None:
                #     pred_points = pred
                #     pred_colors = np.tile(color_norm[int(itemid-1)], (pred.shape[0],1))
                # else:
                #     pred_points = np.vstack((pred_points, pred))
                #     pred_colors = np.vstack((pred_colors, np.tile(color_norm[int(itemid-1)], (pred.shape[0],1))))

                # if world_points is None:
                #     world_points = world_model
                #     world_colors = np.tile(color_norm[int(itemid-1)], (world_model.shape[0],1))
                # else:
                #     world_points = np.vstack((world_points, world_model))
                #     world_colors = np.vstack((world_colors, np.tile(color_norm[int(itemid-1)], (world_model.shape[0],1))))

                # # interesting_points
                # mask_label_roi = mask[rmin:rmax, cmin:cmax]
                # mask_label = np.zeros((img_height, img_width))
                # mask_label[rmin:rmax, cmin:cmax] = mask_label_roi
                # predicted_points, matching_distance_map, matching_distance, background_region = get_matching_score(pred[:,:3], reverse_mask, mask_label, depth)
                # if predicted_points is None:
                #     continue

                # if matching_distance < 0.1:
                #     excluding_pixels = np.where(matching_distance_map == 0)
                #     matching_distance_map[excluding_pixels] += 1
                #     interesting_pixels = np.where(matching_distance_map < 0.01)

                #     interesting_pixels_y = interesting_pixels[1][interesting_pixels[1] - cmin < cmax - cmin]
                #     interesting_pixels_x = interesting_pixels[0][interesting_pixels[0] - rmin < rmax - rmin]

                #     interesting_model = predicted_points[interesting_pixels_x, interesting_pixels_y]
                #     external_dim = np.ones_like(interesting_model[:,0]).reshape(interesting_model[:,0].size, 1)
                #     interesting_model = np.hstack([interesting_model, external_dim])

                #     cnn_img = cnn_img.squeeze()
                #     interesting_cnn_feature = cnn_img[:, interesting_pixels_x - rmin, interesting_pixels_y - cmin].detach().cpu().numpy().T

                #     interesting_model = np.dot(inv(TransE), interesting_model.T).T[:,:3]     # Model points in world coordinate
                #     interesting_model = np.hstack([interesting_model, interesting_cnn_feature])
                #     # print interesting_model[0]


                #     if self.interesting_points[itemid] is None:
                #         if self.interesting_points[0] is None:
                #             self.interesting_points[0] = interesting_model
                #             self.interesting_colors[0] = np.tile(color_norm[int(itemid-1)], (interesting_model.shape[0],1))
                #         self.interesting_points[itemid] = interesting_model
                #         self.interesting_colors[itemid] = np.tile(color_norm[int(itemid-1)], (interesting_model.shape[0],1))
                #     else:
                #         self.interesting_points[itemid] = np.vstack((self.interesting_points[itemid], interesting_model))
                #         self.interesting_colors[itemid] = np.vstack((self.interesting_colors[itemid], np.tile(color_norm[int(itemid-1)], (interesting_model.shape[0],1))))
                #         self.interesting_points[0] = np.vstack((self.interesting_points[0], interesting_model))
                #         self.interesting_colors[0] = np.vstack((self.interesting_colors[0], np.tile(color_norm[int(itemid-1)], (interesting_model.shape[0],1))))
            
            except ZeroDivisionError:
                print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))

        if self.is_first:
            self.is_first = False

        # image_mat0 = draw_img[:,:,0].copy()
        # draw_img[:,:,0] = draw_img[:,:,2]
        # draw_img[:,:,2] = image_mat0
        # self.result_pub.publish(self.br.cv2_to_imgmsg(draw_img))

        cam_model = np.dot(inv(TransE), self.camera_model.T).T
        # world_points = np.vstack((world_points, cam_model))
        # world_colors = np.vstack((world_colors, np.tile(color_norm[int(2)], (cam_model.shape[0],1))))
        world_points = cam_model
        world_colors = np.tile(color_norm[int(2)], (cam_model.shape[0],1))

        # point_cloud = xyzrgb_array_to_pointcloud2(pred_points[:,:3], pred_colors, frame_id="base_link")
        # self.point_cloud_pub.publish(point_cloud)
        world_point_cloud = xyzrgb_array_to_pointcloud2(world_points[:,:3], world_colors, frame_id="map")
        self.world_point_cloud_pub.publish(world_point_cloud)
        # print("world_point_cloud.shape : ", world_points.shape)

        # # Publish interesting points
        # if self.interesting_points[0] is not None:
        #     interesting_point_cloud = xyzrgb_array_to_pointcloud2(self.interesting_points[0][:,:3], self.interesting_colors[0], frame_id="base_scan")
        #     # print("interesting_point_cloud.shape : ", interesting_points[0].shape)
        #     self.interesting_points_pub.publish(interesting_point_cloud)

        # Publish initial points
        if self.first_world_model is not None:
            initial_point_cloud = xyzrgb_array_to_pointcloud2(self.first_world_model[:,:3], self.first_world_model_color, frame_id="map")
            # print("interesting_point_cloud.shape : ", interesting_points[0].shape)
            self.interesting_points_pub.publish(initial_point_cloud)

        # Check projected world model points.
        camera_to_model_points = np.dot(TransE, self.first_world_model.T).T
        # camera_to_model_points = np.dot(self.first_world_model - E_T, inv(E_R))
        x = cam_fx * camera_to_model_points[:,0] / camera_to_model_points[:,2] + cam_cx
        y = cam_fy * camera_to_model_points[:,1] / camera_to_model_points[:,2] + cam_cy

        x_1 = x[x[:] < img_width].T
        y_1 = y[x[:] < img_width].T
        x = x_1[y_1[:] < img_height].T
        y = y_1[y_1[:] < img_height].T

        x_1 = x[x[:] >= 0].T
        y_1 = y[x[:] >= 0].T
        x = x_1[y_1[:] >= 0].T
        y = y_1[y_1[:] >= 0].T

        x = np.vectorize(np.int)(x)
        y = np.vectorize(np.int)(y)
        
        img_projected = np.array(img)[:, :, :3].copy()
        img_projected0 = img_projected[:,:,0].copy()
        img_projected[:,:,0] = img_projected[:,:,2]
        img_projected[:,:,2] = img_projected0
        img_projected[y, x] = color[10]

        self.result_pub.publish(self.br.cv2_to_imgmsg(img_projected))

        self.bPubImgLock = False
        # return

if __name__ == '__main__':
    
    # bc = np.load('/home/user/catkin_ws/src/densefusion_py/interesting_points/scene10/bleach_cleanser.npy')
    # print bc[0]
    # print bc.shape

    rospy.init_node('py_densefusion', anonymous=False)
    df = Densefusion()
    # df.densefusion()
    # df.densefusion_live()
    df.image_pub()

    #py_trees.logging.level = py_trees.logging.Level.DEBUG

    # print("starting..")
    # print('initialize...')

    # rospy.spin()
