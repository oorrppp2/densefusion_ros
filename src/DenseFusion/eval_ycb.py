import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image as PIL_Image
import argparse
import cv2
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import PoseDataset
from network import PoseNet, PoseRefineNet
from transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

from DenseFusion.yolact import Yolact
from collections import defaultdict

from numpy.linalg import inv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--posecnn_model', type=str, default = '',  help='resume PoseCNN model')

# yolact args
parser.add_argument('--trained_model',
                    default='/home/user/catkin_ws/src/densefusion_ros/src/DenseFusion/weights/yolact_resnet50_204_90000.pth', type=str,
                    help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
# parser.add_argument('--cross_class_nms', default=False, type=str2bool,
#                     help='Whether compute NMS cross-class or per-class.')
# parser.add_argument('--display_masks', default=True, type=str2bool,
#                     help='Whether or not to display masks over bounding boxes')
# parser.add_argument('--display_bboxes', default=True, type=str2bool,
#                     help='Whether or not to display bboxes around masks')
# parser.add_argument('--display_text', default=True, type=str2bool,
#                     help='Whether or not to display text (class [score])')
# parser.add_argument('--display_scores', default=True, type=str2bool,
#                     help='Whether or not to display scores in addition to classes')
parser.add_argument('--display', dest='display', action='store_true',
                    help='Display qualitative results instead of quantitative ones.')
parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                    help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                    help='In quantitative mode, the file to save detections before calculating mAP.')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='If display not set, this resumes mAP calculations from the ap_data_file.')
parser.add_argument('--max_images', default=-1, type=int,
                    help='The maximum number of images from the dataset to consider. Use -1 for all.')
parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                    help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                    help='The output file for coco bbox results if --coco_results is set.')
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                    help='The output file for coco mask results if --coco_results is set.')
parser.add_argument('--config', default='yolact_ycb_config',
                    help='The config object to use.')
parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                    help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
parser.add_argument('--web_det_path', default='web/dets/', type=str,
                    help='If output_web_json is set, this is the path to dump detections into.')
parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                    help='Do not output the status bar. This is useful for when piping to a file.')
parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                    help='Equivalent to running display mode but without displaying an image.')
parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                    help='Do not sort images by hashed image ID.')
parser.add_argument('--seed', default=None, type=int,
                    help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                    help='Outputs stuff for scripts/compute_mask.py.')
parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                    help='Do not crop output masks with the predicted bounding box.')
parser.add_argument('--image', default=None, type=str,
                    help='A path to an image to use for display.')
parser.add_argument('--images', default=None, type=str,
                    help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
parser.add_argument('--video', default=None, type=str,
                    help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
parser.add_argument('--video_multiframe', default=1, type=int,
                    help='The number of frames to evaluate in parallel to make videos play at higher fps.')
parser.add_argument('--score_threshold', default=0, type=float,
                    help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
parser.add_argument('--dataset', default='ycb_dataset', type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                    help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                    help='When displaying / saving video, draw the FPS on the frame')
parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                    help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                    shuffle=False,
                    benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                    display_fps=False,
                    emulate_playback=False)


opt = parser.parse_args()

dataset_root_dir = '/media/user/ssd_1TB/YCB_dataset'
model_dir = '/home/user/catkin_ws/src/densefusion_ros/src/DenseFusion/trained_models/ycb/pose_model_26_0.012863246640872631.pth'
refine_model_dir = '/home/user/catkin_ws/src/densefusion_ros/src/DenseFusion/trained_models/ycb/pose_refine_model_69_0.009449292959118935.pth'

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 0.0001

K = [[cam_fx , 0 , cam_cx],
     [0, cam_fy, cam_cy],
     [0, 0, 1]]

num_obj = 21
img_width = 640
img_height = 480
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = '/home/user/catkin_ws/src/densefusion_ros/src/DenseFusion/datasets/ycb/dataset_config'
ycb_toolbox_dir = '/home/user/catkin_ws/src/densefusion_ros/src/DenseFusion/YCB_Video_toolbox'

color = [[16, 122, 180], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0],
              [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128],
              [251, 194, 44], [240, 20, 134], [160, 103, 173], [70, 163, 210], [140, 227, 61],
              [128, 128, 0], [128, 0, 128], [0, 128, 128], [196, 235, 77], [77, 235, 255], [0, 0, 64]]
color_norm = [[0.125, 0.7533, 0.894], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0],
              [1, 0, 1], [0, 1, 1], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
              [1, 0.6789, 0.2349], [1, 0.0165, 0.441], [0.49, 0.36, 0.65], [0.12, 0.5, 0.789], [0.498, 0.678, 0.16],
              [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.59, 0.92, 0.24], [0.24, 0.89, 1], [0, 0, 0.3]]

class_name = ['master_chef_can', 'cracker_box', 'sugar_box', 'tomato_soup_can', 'mustard_bottle', 'tuna_fish_can', 'pudding_box', 'gelatin_box',
              'potted_meat_can', 'banana', 'pitcher_base', 'bleach_cleanser', 'bowl', 'mug', 'power_drill', 'wood_block', 'scissors', 'large_marker',
              'large_clamp', 'extra_large_clamp', 'foam_brick']

def get_bbox(posecnn_rois, idx):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_height:
        delt = rmax - img_height
        rmax = img_height
        rmin -= delt
    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    return rmin, rmax, cmin, cmax

estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(model_dir))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(refine_model_dir))
refiner.eval()

# yolact = Yolact()
# yolact.load_weights(opt.trained_model)
# yolact.eval()
# yolact.cuda()

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# yolact.detect.use_fast_nms = True
# yolact.detect.use_cross_class_nms = False

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
draw_cld = {}
corners = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/textured.obj'.format(dataset_root_dir, class_input))
    xyz_input_file = open('{0}/models/{1}/points.xyz'.format(dataset_root_dir, class_input))
    cld[class_id] = []
    draw_cld[class_id] = []
    x = []
    y = []
    z = []
    # while 1:
    #     input_line = input_file.readline()
    #     if not input_line:
    #         break
    #     input_line = input_line[:-1]
    #     input_line = input_line.split(' ')

    #     if input_line[0] == 'v':
    #         cld[class_id].append([float(input_line[1]), float(input_line[2]), float(input_line[3])])
    #         # x.append(float(input_line[1]))
    #         # y.append(float(input_line[2]))
    #         # z.append(float(input_line[3]))
    #     if input_line[0] == 'vt' or input_line[0] == 'vn':
    #         break
    while 1:
        input_line = xyz_input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')

        draw_cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        x.append(float(input_line[0]))
        y.append(float(input_line[1]))
        z.append(float(input_line[2]))

    input_file.close()
    xyz_input_file.close()
    cld[class_id] = np.array(cld[class_id])
    draw_cld[class_id] = np.array(draw_cld[class_id])

    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    z_max = np.max(z)
    z_min = np.min(z)

    corners[class_id] = np.array([[x_min, y_min, z_min],
                                [x_max, y_min, z_min],
                                [x_max, y_max, z_min],
                                [x_min, y_max, z_min],

                                [x_min, y_min, z_max],
                                [x_max, y_min, z_max],
                                [x_max, y_max, z_max],
                                [x_min, y_max, z_max]])
    class_id += 1
cld = draw_cld

def draw_object(itemid, my_r, my_t, img):
    model_points = draw_cld[itemid]
    my_r = quaternion_matrix(my_r)[:3, :3]
    pred = np.dot(model_points, my_r.T) + my_t

    corner = corners[itemid]
    pred_box = np.dot(corner, my_r.T) + my_t
    transposed_pred_box = pred_box.T
    pred_box = transposed_pred_box/transposed_pred_box[2,:]
    pred_box_pixel = np.matmul(K, pred_box)
    # pred_box_pixel = K @ pred_box
    pred_box_pixel = pred_box_pixel.astype(np.int64)

    cv2.line(img, (pred_box_pixel[0, 0], pred_box_pixel[1, 0]),
           (pred_box_pixel[0, 1], pred_box_pixel[1, 1]), (255,0,0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 1], pred_box_pixel[1, 1]),
           (pred_box_pixel[0, 2], pred_box_pixel[1, 2]), (255,0,0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 2], pred_box_pixel[1, 2]),
           (pred_box_pixel[0, 3], pred_box_pixel[1, 3]), (255,0,0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 3], pred_box_pixel[1, 3]),
           (pred_box_pixel[0, 0], pred_box_pixel[1, 0]), (255,0,0), 2, lineType=cv2.LINE_AA)

    cv2.line(img, (pred_box_pixel[0, 4], pred_box_pixel[1, 4]),
           (pred_box_pixel[0, 5], pred_box_pixel[1, 5]), (255,0,0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 5], pred_box_pixel[1, 5]),
           (pred_box_pixel[0, 6], pred_box_pixel[1, 6]), (255,0,0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 6], pred_box_pixel[1, 6]),
           (pred_box_pixel[0, 7], pred_box_pixel[1, 7]), (255,0,0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 7], pred_box_pixel[1, 7]),
           (pred_box_pixel[0, 4], pred_box_pixel[1, 4]), (255,0,0), 2, lineType=cv2.LINE_AA)

    cv2.line(img, (pred_box_pixel[0, 0], pred_box_pixel[1, 0]),
           (pred_box_pixel[0, 4], pred_box_pixel[1, 4]), (255,0,0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 1], pred_box_pixel[1, 1]),
           (pred_box_pixel[0, 5], pred_box_pixel[1, 5]), (255,0,0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 2], pred_box_pixel[1, 2]),
           (pred_box_pixel[0, 6], pred_box_pixel[1, 6]), (255,0,0), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (pred_box_pixel[0, 3], pred_box_pixel[1, 3]),
           (pred_box_pixel[0, 7], pred_box_pixel[1, 7]), (255,0,0), 2, lineType=cv2.LINE_AA)

    transposed_pred = pred.T
    pred = transposed_pred/transposed_pred[2,:]
    pred_pixel = np.matmul(K, pred)
    # K @ pred
    pred_pixel = pred_pixel.astype(np.int64)

    _, cols = pred_pixel.shape
    del_list = []
    for i in range(cols):
        if pred_pixel[0,i] >= img_width or pred_pixel[1,i] >= img_height or \
                pred_pixel[0, i] < 0 or pred_pixel[1, i] < 0 :
            del_list.append(i)
    pred_pixel = np.delete(pred_pixel, del_list, axis=1)

    img[pred_pixel[1,:], pred_pixel[0,:]] = color[int(itemid-1)]
    return img


def get_matching_score(pred, reverse_mask, mask_label, depth):
    # if dis == "inf":
    #     return 0

    predicted_depth = np.zeros((img_height, img_width, 3))
    target_object_region = mask_label
    depth_array = pred[:,2]

    transposed_pred = pred.T
    pred = transposed_pred/transposed_pred[2,:]
    # pred_pixel = K @ pred
    pred_pixel = np.dot(K, pred)
    pred_pixel = pred_pixel.astype(np.int64)
    # print(pred_pixel)
    depth_array = np.squeeze(depth_array)
    # print(depth_array.shape)

    sorted_arg = np.argsort(-depth_array)
    # sorted_arg = np.squeeze(sorted_arg)
    # print(sorted_arg)
    depth_array = depth_array[0, sorted_arg]
    pred_pixel = pred_pixel[:, sorted_arg]
    # print(pred_pixel)
    # print(pred_pixel.shape)
    pred_pixel = np.squeeze(pred_pixel, axis=1)
    # print(pred_pixel.shape)
    # depth_array = np.squeeze(depth_array)
    # print(depth_array.shape)
    _, cols = pred_pixel.shape

    # print(pred_pixel)
    del_list = []
    for i in range(cols):
        if pred_pixel[0,i] >= img_width or pred_pixel[1,i] >= img_height or \
                pred_pixel[0, i] < 0 or pred_pixel[1, i] < 0 :
            del_list.append(i)
    pred_pixel = np.delete(pred_pixel, del_list, axis=1)
    depth_array = np.delete(depth_array, del_list, axis=1)

    if pred_pixel.size == 0:
        return None, None, None, None
    pred = np.matmul(inv(K), pred_pixel)
    pred[2, :] = depth_array

    # print "pred shape : ", pred.shape
    # print("pred nonzero : ", pred.nonzero())
    # print("depth_array shape : ", depth_array.shape)

    # predicted_depth[pred_pixel[1,:], pred_pixel[0,:]] = depth_array.T
    # predicted_depth[pred_pixel[1,:], pred_pixel[0,:]] = depth_array
    
    # print(predicted_depth.shape)
    # print(predicted_depth[pred_pixel[1,:], pred_pixel[0,:], 0].shape)
    # print(pred.T.shape)
    predicted_depth[pred_pixel[1,:], pred_pixel[0,:]] = pred.T.reshape(1,-1,3)
    # print(predicted_depth.nonzero())
    # print(predicted_depth[pred_pixel[1,:], pred_pixel[0,:]])
    # print(pred.T.reshape(1,-1,3))

    self_occlusion = ma.getmaskarray(ma.masked_equal(depth, 0))
    target_object_region = ma.getmaskarray(ma.masked_not_equal(target_object_region, 0))

    background_region = np.zeros((img_height, img_width, 3))
    background_region[reverse_mask] = 1             # Background region
    background_region[target_object_region] = 1     # Target object region
    background_region[self_occlusion] = 0           # Target object region
    #
    # non_zero_predicted_depth = ma.getmaskarray(ma.masked_not_equal(predicted_depth, 0))
    # masked_depth = (depth * cam_scale) * target_object_region * non_zero_predicted_depth
    # masked_depth = (depth * cam_scale) * target_object_region
    non_zero_predicted_depth = ma.getmaskarray(ma.masked_not_equal(predicted_depth, 0))[:,:,0]
    masked_depth = (depth * cam_scale) * target_object_region * non_zero_predicted_depth
    #
    predicted_depth_mask = background_region * predicted_depth
    # return predicted_depth_mask, background_region

    # matching_score_map = abs(predicted_depth_mask - masked_depth)
    matching_distance_map = abs(predicted_depth_mask[:,:,2] - masked_depth)
    # print "matching_distance_map sum : " , np.sum(matching_distance_map)
    # matching_score = torch.dist(torch.Tensor(predicted_depth_mask[:,:,2]), torch.Tensor(masked_depth), p=2)
    matching_distance_flatten = matching_distance_map.flatten()
    matching_distance_array = matching_distance_flatten[matching_distance_flatten.nonzero()]
    # matching_distance = np.sum(matching_distance_array) / len(matching_distance_array)
    matching_distance = np.sum(matching_distance_array) / len(matching_distance_array)

    print "matching_distance : ", matching_distance

    return predicted_depth, matching_distance_map, matching_distance, background_region
    # print("dist : ", dist)

    # cv2.imshow("predicted_depth_mask", predicted_depth_mask)
    # cv2.imshow("predicted_depth", predicted_depth)
    # cv2.imshow("masked_depth", masked_depth)
    # cv2.imshow("matching_score", matching_score)
    # cv2.waitKey(0)

