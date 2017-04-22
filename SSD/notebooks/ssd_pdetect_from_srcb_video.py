import tensorflow as tf
import cv2

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

slim = tf.contrib.slim

import sys
sys.path.append('../')

import time
import os

from nets import ssd_vgg_300, ssd_vgg_512, np_methods
from preprocessing import ssd_vgg_preprocessing

sess = tf.Session()

# net_shape = (512, 1024)
net_shape = (600, 1200)
data_format = 'NCHW'    # GPU run
# data_format = 'NHWC'    # CPU BiasOp only supports NHWC
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
# ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
# # to test self trained model
ckpt_filename = '../logs/'
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


def bboxes_draw_on_img(img, classes, scores, bboxes, color, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)

# read video clip
# video_file = './demo/1_er_1920x960.MP4'
video_file = '../demo/b.webm'
# video_file = '../demo/TY_20170311_18_30_front.mp4'
print("loading {}".format(video_file))
cap = cv2.VideoCapture(video_file)

# color for boxes
color = (0, 69, 255)    # BGR, orange red

output_shape = (480, 960)

frame_counter = 0
img_output_folder = './img_out_tmp/'
def clear_img_tmp_folder(path_to_folder):
    for file in os.listdir(path_to_folder):
        file_path = os.path.join(path_to_folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print e
clear_img_tmp_folder(img_output_folder)

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame_counter += 1
        # resize
        img = cv2.resize(frame, net_shape[::-1], interpolation=cv2.INTER_CUBIC)
        start = time.time()
        rclasses, rscores, rbboxes = process_image(img, net_shape=net_shape)
        end = time.time()
        print('Frame {} time elapsed to process one {} img: {}'.format(frame_counter, net_shape, end-start))
        person_select_indicator = (rclasses == 15)
        rclasses = rclasses[person_select_indicator]
        rscores = rscores[person_select_indicator]
        rbboxes = rbboxes[person_select_indicator]
        bboxes_draw_on_img(img, rclasses, rscores, rbboxes, color=color)
        # save
        img = cv2.resize(img, output_shape[::-1], interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('{}frame_{:05d}.png'.format(img_output_folder, frame_counter), img)
    else:
        break
print('Total # of frames: {}'.format(frame_counter))

cap.release()
cv2.destroyAllWindows()
