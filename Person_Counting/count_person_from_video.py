import cv2

import time
import numpy as np
import os
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

slim = tf.contrib.slim

import sys
sys.path.append('../SSD')
sys.path.append('..')

from SSD.nets import ssd_vgg_512, np_methods
from SSD.preprocessing import ssd_vgg_preprocessing

from sort.sort import Sort       # for tracking

sess = tf.Session()

net_shape = (512, 1024)     # from origin (1080, 1960), for 512 or 300 SSD
data_format = 'NCHW'        # for GPU
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

ckpt_filename = '../SSD/checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

ssd_anchors = ssd_net.anchors(net_shape)

def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = sess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# video clip
# date = '20170304'       # sat - side
# date = '20170310'       # fri - front
# date = '20170419'       # fri - front
date = '20170420'       # sat
cam_pose = 'front'      # 'side' or 'front'
total_pcount_each_minute = np.zeros((12, 60), dtype=np.int32)       # 12 hours from 10am to 22pm

# prepare id tracker
mot_tracker = Sort(max_age=10, min_hits=3)

for hour in np.arange(10,22):
    for minute in np.arange(60):
        print("loading ../datasets/TongYing/{}/{}/{:02d}/{:02d}.mp4".format(cam_pose, date, hour, minute))
        cap = cv2.VideoCapture('../datasets/TongYing/{}/{}/{:02d}/{:02d}.mp4'.format(cam_pose, date, hour, minute))

        mot_tracker.update([])      # just in case the first file does not exist

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                # resize
                img = cv2.resize(frame, net_shape[::-1], interpolation=cv2.INTER_CUBIC)
                # start = time.time()
                rclasses, rscores, rbboxes = process_image(img, net_shape=net_shape)
                # end = time.time()
                # # debug
                # print('Time elapsed to process one {} img: {:.03f} sec'.format(net_shape, end-start))

                person_select_indicator = (rclasses == 15)  # pedestrians only
                rclasses = rclasses[person_select_indicator]
                rscores = rscores[person_select_indicator]      # confidence
                rbboxes = rbboxes[person_select_indicator]

                if rbboxes.__len__() > 0:
                    mot_tracker.update(np.hstack((rbboxes, rscores[:, np.newaxis])))
            else:
                break
        # update when all frames in one minute have been scanned
        if mot_tracker.trackers.__len__() > 0:
            total_pcount_each_minute[hour - 10][minute] = mot_tracker.trackers[0].count

cap.release()
cv2.destroyAllWindows()

np.savetxt('outputs/id_accumulated_counter_{}_{}_SSD512x1024_sort_ma10_mh3.txt'.format(date, cam_pose), np.array(total_pcount_each_minute), fmt='%d', delimiter=',')
