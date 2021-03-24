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

# from yolact.data.coco import COCODetection, get_label_map, MEANS, COLORS, cfg, set_cfg, set_dataset

from DenseFusion._init_paths import *
from DenseFusion.eval_ycb_predict_registered_points import *

from std_msgs.msg import Empty, String, Bool, Header, Float64
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge

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
        self.image_test_pub = rospy.Publisher("/camera/color/image_test", Image, queue_size=1)
        self.depth_raw_pub = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)
        self.point_cloud_pub = rospy.Publisher("/point_cloud_densefusion", PointCloud2, queue_size=1)
        self.world_point_cloud_pub = rospy.Publisher("/world_point_cloud_densefusion", PointCloud2, queue_size=1)
        self.interesting_points_pub = rospy.Publisher("/interesting_points_densefusion", PointCloud2, queue_size=1)

        self.image_raw_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.handle_image_sub)
        self.depth_raw_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.handle_depth_sub)
        self.br = CvBridge()

        self.image = None
        self.depth = None


        # print self.interesting_points[0].shape

        # self.densefusion

    def handle_image_sub(self, msg):
        self.image = self.br.imgmsg_to_cv2(msg)
        print("image shape : ", self.image.shape)
    def handle_depth_sub(self, msg):
        self.depth = self.br.imgmsg_to_cv2(msg)
        print("depth shape : ", self.depth.shape)

    def densefusion_live(self):
        return

    def check_loaded_arrays(self):
        interesting_points = {}
        interesting_colors = {}
        interesting_points[0] = None
        interesting_colors[0] = None
        for i in range(1,22):
            if os.path.isfile('/home/user/catkin_ws/src/densefusion_py/interesting_points/scene10/'+class_name[i-1]+'.npy'):
                interesting_points[i] = np.load('/home/user/catkin_ws/src/densefusion_py/interesting_points/scene10/'+class_name[i-1]+'.npy')
                interesting_colors[i] = np.tile(color_norm[int(i-1)], (interesting_points[i].shape[0],1))
                if interesting_points[0] is None:
                    interesting_points[0] = np.load('/home/user/catkin_ws/src/densefusion_py/interesting_points/scene10/'+class_name[i-1]+'.npy')
                    interesting_colors[0] = np.tile(color_norm[int(i-1)], (interesting_points[i].shape[0],1))
                else:
                    interesting_points[0] = np.vstack((interesting_points[0], interesting_points[i]))
                    interesting_colors[0] = np.vstack((interesting_colors[0], np.tile(color_norm[int(i-1)], (interesting_points[i].shape[0],1))))
                print class_name[i-1] + ' loaded'


        # Publish interesting points
        # while interesting_points[0] is not None:
        #     interesting_point_cloud = xyzrgb_array_to_pointcloud2(interesting_points[0][:,:3], interesting_colors[0], frame_id="base_scan")
        #     print("interesting_point_cloud.shape : ", interesting_points[0].shape)
        #     self.interesting_points_pub.publish(interesting_point_cloud)

    def densefusion(self):

        interesting_points = {}
        interesting_colors = {}
        interesting_points[0] = None
        interesting_colors[0] = None
        for i in range(1,22):
            if os.path.isfile('/home/user/catkin_ws/src/densefusion_py/interesting_points/scene10/'+class_name[i-1]+'.npy'):
                interesting_points[i] = np.load('/home/user/catkin_ws/src/densefusion_py/interesting_points/scene10/'+class_name[i-1]+'.npy')
                interesting_colors[i] = np.tile(color_norm[int(i-1)], (interesting_points[i].shape[0],1))
                if interesting_points[0] is None:
                    interesting_points[0] = np.load('/home/user/catkin_ws/src/densefusion_py/interesting_points/scene10/'+class_name[i-1]+'.npy')
                    interesting_colors[0] = np.tile(color_norm[int(i-1)], (interesting_points[i].shape[0],1))
                else:
                    interesting_points[0] = np.vstack((interesting_points[0], interesting_points[i]))
                    interesting_colors[0] = np.vstack((interesting_colors[0], np.tile(color_norm[int(i-1)], (interesting_points[i].shape[0],1))))
                print class_name[i-1] + ' loaded'

        # for now in range(1395, 2949):
        # for now in range(1500, 2949):
        # for now in range(1665, 2949):
        # for now in range(1899, 2087):   # Scene 9
        for now in range(2088, 2375):   # Scene 10
        # for now in range(0, 394):   # Scene1 for test.
        # for now in range(2600, 2949):   # Scene1 for test.
            if not rospy.is_shutdown():
                img = PIL_Image.open('{0}/{1}-color.png'.format(dataset_root_dir, testlist[now]))
                depth = np.array(PIL_Image.open('{0}/{1}-depth.png'.format(dataset_root_dir, testlist[now])))
                posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
                label = np.array(posecnn_meta['labels'])
                posecnn_rois = np.array(posecnn_meta['rois'])

                camera_meta = scio.loadmat('{0}/{1}-meta.mat'.format(dataset_root_dir, testlist[now]))
                extrinsic = np.array(camera_meta['rotation_translation_matrix'])
                # print(extrinsic)
                # extrinsic rotation
                E_R = extrinsic[:,:3]
                E_T = extrinsic[:,3]
                E_T = E_T.reshape(3,1)
                external_row = np.matrix([0, 0, 0, 1])
                TransE = np.hstack((E_R, E_T))
                TransE = np.vstack((TransE, external_row))
                # print(TransE)
                # print(inv(TransE))

                pred_points = None
                pred_colors = None

                world_points = None
                world_colors = None

                # interesting_points = None
                # interesting_colors = None

                pub_img = img.copy()
                pub_img = np.array(pub_img)[:, :, :3]
                pub_img.astype(np.float32)
                pub_depth = depth.copy()
                pub_depth = np.array(pub_depth)
                pub_depth = pub_depth.astype(np.float)
                pub_depth *= cam_scale
                # self.image_raw_pub.publish(self.br.cv2_to_imgmsg(pub_img))
                # self.depth_raw_pub.publish(self.br.cv2_to_imgmsg(pub_depth))

                lst = posecnn_rois[:, 1:2].flatten()

                reverse_mask = ma.getmaskarray(ma.masked_equal(label, 0))

                draw_img = img.copy()
                draw_img = np.array(draw_img)[:, :, :3]
                draw_img.astype(np.float32)
                # image_mat0 = draw_img[:,:,0].copy()
                # draw_img[:,:,0] = draw_img[:,:,2]
                # draw_img[:,:,2] = image_mat0
                matching_score_zeros_map = np.zeros((img_height, img_width, 3))
                
                for idx in range(len(lst)):
                    itemid = int(lst[idx])
                    try:
                        print " *** " , class_name[itemid -1] , " ***"
                        # if interesting_points[itemid].size 
                        rmin, rmax, cmin, cmax = get_bbox(posecnn_rois, idx)

                        np.random.shuffle(interesting_points[itemid])
                        print interesting_points[itemid][:num_points].shape
                        cloud = interesting_points[itemid][:num_points][:,:3]
                        external_dim = np.ones_like(cloud[:,0]).reshape(cloud[:,0].size, 1)
                        cloud = np.hstack([cloud, external_dim])
                        cloud = np.dot(TransE, cloud.T).T[:,:3]

                        cloud = torch.from_numpy(cloud.astype(np.float32))
                        cloud = Variable(cloud).cuda()
                        index = torch.LongTensor([itemid - 1])
                        # choose = Variable(choose).cuda()
                        index = Variable(index).cuda()
                        cloud = cloud.view(1, num_points, 3)

                        print cloud.shape
                        cnn_feature = interesting_points[itemid][:num_points][:,3:].T
                        cnn_feature = torch.from_numpy(cnn_feature.astype(np.float32))
                        cnn_feature = Variable(cnn_feature).cuda()
                        cnn_feature = cnn_feature.view(1, 32, num_points)
                        print cnn_feature.shape

                        pred_r, pred_t, pred_c = estimator(cnn_feature, cloud, index)
                        # print(pred_c)
                        # cloud = cloud.squeeze()
                        # x = cam_fx * cloud[:,0] / cloud[:,2] + cam_cx
                        # y = cam_fy * cloud[:,1] / cloud[:,2] + cam_cy
                        # for i in range(len(x)):
                        #     x[i] = int(x[i])
                        # for i in range(len(y)):
                        #     y[i] = int(y[i])
                        
                        # img_projected = np.array(img)[:, :, :3].copy()
                        # img_projected0 = img_projected[:,:,0].copy()
                        # img_projected[:,:,0] = img_projected[:,:,2]
                        # img_projected[:,:,2] = img_projected0
                        
                        # for i in range(len(x)):
                        #     if y[i] < img_height and x[i] < img_width and y[i] >= 0 and x[i] >= 0:
                        #         img_projected[int(y[i])][int(x[i])] = color[10]
                        # self.result_pub.publish(self.br.cv2_to_imgmsg(img_projected))
                        # rospy.sleep(5)

                        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

                        pred_c = pred_c.view(bs, num_points)
                        how_max, which_max = torch.max(pred_c, 1)
                        pred_t = pred_t.view(bs * num_points, 1, 3)
                        points = cloud.view(bs * num_points, 1, 3)

                        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                        my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                        my_pred = np.append(my_r, my_t)

                        # continue

                        # for ite in range(0, iteration):
                        #     T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                        #     my_mat = quaternion_matrix(my_r)
                        #     R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                        #     my_mat[0:3, 3] = my_t
                            
                        #     new_cloud = torch.bmm((cloud - T), R).contiguous()
                        #     pred_r, pred_t = refiner(new_cloud, emb, index)
                        #     pred_r = pred_r.view(1, 1, -1)
                        #     pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                        #     my_r_2 = pred_r.view(-1).cpu().data.numpy()
                        #     my_t_2 = pred_t.view(-1).cpu().data.numpy()
                        #     my_mat_2 = quaternion_matrix(my_r_2)

                        #     my_mat_2[0:3, 3] = my_t_2

                        #     my_mat_final = np.dot(my_mat, my_mat_2)
                        #     my_r_final = copy.deepcopy(my_mat_final)
                        #     my_r_final[0:3, 3] = 0
                        #     my_r_final = quaternion_from_matrix(my_r_final, True)
                        #     my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                        #     my_pred = np.append(my_r_final, my_t_final)
                        #     my_r = my_r_final
                        #     my_t = my_t_final


                        draw_img = draw_object(int(itemid), my_r, my_t, draw_img)

                        # # TransM : camera to model transform matrix
                        # M_R = quaternion_matrix(my_r)[:3, :3]
                        # M_T = my_t
                        # M_T = M_T.reshape(3,1)
                        # TransM = np.hstack((M_R, M_T))
                        # TransM = np.vstack((TransM, external_row))
                        # trans_world_to_model = np.dot(TransE, TransM)

                        # model_points = cld[itemid]
                        # external_dim = np.ones_like(model_points[:,0]).reshape(model_points[:,0].size, 1)
                        # model_points = np.hstack([model_points, external_dim])

                        # pred = np.dot(TransM, model_points.T).T         # Model points in camera coordinate
                        # world_model = np.dot(inv(TransE), pred.T).T     # Model points in world coordinate

                        # # Check camera trajectory using projection.
                        # if is_first:
                        #     first_world_model = world_model
                        #     is_first = False

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

                    except ZeroDivisionError:
                        print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
                    except KeyError:
                        print "Key error occur!"

                image_mat0 = draw_img[:,:,0].copy()
                draw_img[:,:,0] = draw_img[:,:,2]
                draw_img[:,:,2] = image_mat0
                self.result_pub.publish(self.br.cv2_to_imgmsg(draw_img))

                # point_cloud = xyzrgb_array_to_pointcloud2(pred_points[:,:3], pred_colors, frame_id="base_scan")
                # self.point_cloud_pub.publish(point_cloud)
                # world_point_cloud = xyzrgb_array_to_pointcloud2(world_points[:,:3], world_colors, frame_id="base_scan")
                # self.world_point_cloud_pub.publish(world_point_cloud)

                # Publish interesting points
                if interesting_points[0] is not None:
                    interesting_point_cloud = xyzrgb_array_to_pointcloud2(interesting_points[0][:,:3], interesting_colors[0], frame_id="base_scan")
                    self.interesting_points_pub.publish(interesting_point_cloud)


                print("Finish No.{0} keyframe".format(now))
            # interesting_points[i]

if __name__ == '__main__':
    
    # bc = np.load('/home/user/catkin_ws/src/densefusion_py/interesting_points/scene10/bleach_cleanser.npy')
    # print bc[0]
    # print bc.shape

    rospy.init_node('py_densefusion', anonymous=False)
    df = Densefusion()
    # df.check_loaded_arrays()
    df.densefusion()
    # df.densefusion_live()

    #py_trees.logging.level = py_trees.logging.Level.DEBUG

    # print("starting..")
    # print('initialize...')

    rospy.spin()
