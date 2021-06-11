# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import os
import sys
import time
import argparse
import numpy as np

# Computer Vision
import cv2
from scipy import ndimage
from skimage.transform import resize

# Visualization
import matplotlib.pyplot as plt

plasma = plt.get_cmap('plasma')
greys = plt.get_cmap('Greys')

# Argument Parser
parser = argparse.ArgumentParser(description='BTS Live 3D')
parser.add_argument('--model_name',      type=str,   help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder',         type=str,   help='type of encoder, densenet121_bts or densenet161_bts', default='densenet161_bts')
parser.add_argument('--max_depth',       type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str,   help='path to a checkpoint to load', required=True)
parser.add_argument('--input_height',    type=int,   help='input height', default=480)
parser.add_argument('--input_width',     type=int,   help='input width',  default=640)

args = parser.parse_args()

model_dir = os.path.join("./models", args.model_name)

sys.path.append(model_dir)
for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val

# Image shapes
height_rgb, width_rgb = 480, 640
height_depth, width_depth = height_rgb, width_rgb
height_rgb = height_rgb

import tensorflow as tf

global graph
graph = tf.get_default_graph()

global sess
# SESSION
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

global image
image = tf.placeholder(tf.float32, [1, 416, 576, 3])


class BTSDepthPredictor():
    def __init__(self):
        self.model = self._load_model()
        print('Model loaded.')

    def _load_model(self):
        params = bts_parameters(
            encoder=args.encoder,
            height=args.input_height,
            width=args.input_width,
            batch_size=None,
            dataset=None,
            max_depth=args.max_depth,
            num_gpus=None,
            num_threads=None,
            num_epochs=None,
        )

        model = BtsModel(params, 'test', image, None, focal=focals, bn_training=False)

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # SAVER
        train_saver = tf.train.Saver()

        with tf.device('/cpu:0'):
            restore_path = args.checkpoint_path
            # RESTORE
            train_saver.restore(sess, restore_path)

        return model

    def predict(self, rgb_frame):
        # image rgb8 format for input
        if self.model:
            input_image = rgb_frame[:, :, :3].astype(np.float32)

            # Normalize image
            input_image[:, :, 0] = (input_image[:, :, 0] - 123.68) * 0.017
            input_image[:, :, 1] = (input_image[:, :, 1] - 116.78) * 0.017
            input_image[:, :, 2] = (input_image[:, :, 2] - 103.94) * 0.017

            input_image_cropped = input_image[32:-1 - 31, 32:-1 - 31, :]

            input_images = np.expand_dims(input_image_cropped, axis=0)
            
            with graph.as_default():
                depth_cropped = sess.run([self.model.depth_est], feed_dict={image: input_images})
            
            raw_depth = np.zeros((480, 640), dtype=np.float32)
            raw_depth[32:-1-31, 32:-1-31] = depth_cropped[0].squeeze() / args.max_depth
            gray_log_depth = (greys(np.log10(depth * args.max_depth))[:, :, :3] * 255).astype('uint8')
            return raw_depth, gray_log_depth


if __name__ == '__main__':
    focals = tf.constant([518.8579])

    # Intrinsic parameters for your own webcam/camera
    camera_matrix = np.zeros(shape=(3, 3))
    camera_matrix[0, 0] = 5.4765313594010649e+02
    camera_matrix[0, 2] = 3.2516069906172453e+02
    camera_matrix[1, 1] = 5.4801781476172562e+02
    camera_matrix[1, 2] = 2.4794113960783835e+02
    camera_matrix[2, 2] = 1
    dist_coeffs = np.array(
        [3.7230261423972011e-02, -1.6171708069773008e-01, -3.5260752900266357e-04, 1.7161234226767313e-04,
         1.0192711400840315e-01])

    # Parameters for a model trained on NYU Depth V2
    new_camera_matrix = np.zeros(shape=(3, 3))
    new_camera_matrix[0, 0] = 518.8579
    new_camera_matrix[0, 2] = 320
    new_camera_matrix[1, 1] = 518.8579
    new_camera_matrix[1, 2] = 240
    new_camera_matrix[2, 2] = 1

    R = np.identity(3, dtype=np.float)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R,
                                             new_camera_matrix, (640, 480), cv2.CV_32FC1)

    frame_bgr = cv2.imread("image")
    frame_bgr = cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    bts_predictor = BTSDepthPredictor()
    depth, gray_log_depth = bts_predictor.predict(frame)
    plt.imshow(gray_log_depth, cmap="gray")
    plt.show()
    plt.imshow(depth, cmap="inferno")
    plt.show()


