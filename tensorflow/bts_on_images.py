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

from glob import glob
import pathlib
import sys
import argparse
import numpy as np

import cv2
import tensorflow as tf

# Get rid of Tensorflow warnings
tf.logging.set_verbosity(tf.logging.ERROR)

global graph
graph = tf.get_default_graph()

global sess
# SESSION
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction=0.60
sess = tf.Session(config=config)


def build_parser():
    parser = argparse.ArgumentParser(description='BTS on images')
    parser.add_argument('images_folder')
    parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
    parser.add_argument('--encoder', type=str, help='type of encoder, densenet121_bts or densenet161_bts',
                        default='densenet161_bts')
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', required=True)
    parser.add_argument('--save_depth_maps', action="store_true", help='Save depth maps from the neural network.')
    parser.add_argument('--recursive_search', action="store_true", help='Considers the IMAGES_FOLDER arg as a tree root.')
    parser.add_argument('--upscaled', action="store_true", help='Upscales the resulting depths to the original image size(s).')
    return parser


def write_array(array, path):
    """
    From COLMAP's utils.
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    import struct

    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(endian_character + format_char_sequence, *data_list)
        fid.write(byte_data)


class BTSDepthPredictor:
    def __init__(self, encoder, max_depth, focal_length_px, upscaling=True):
        self._encoder = encoder
        self._input_width = 576
        self._input_height = 416
        self._input_channels = 3
        self._max_depth = max_depth
        self._do_upscaling = upscaling

        self._image_tensor = tf.placeholder(tf.float32,
                                            [1,
                                             self._input_width,
                                             self._input_height,
                                             self._input_channels])
        self._model = self._load_model(focal_length_px)
        print('Model loaded.')

    def _load_model(self, focal_length_px):
        params = bts_parameters(
            encoder=self._encoder,
            height=self._input_height,
            width=self._input_width,
            batch_size=None,
            dataset=None,
            max_depth=self._max_depth,
            num_gpus=None,
            num_threads=None,
            num_epochs=None,
        )
        model = BtsModel(params, 'test', self._image_tensor, None, focal=focal_length_px, bn_training=False)

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # SAVER
        train_saver = tf.train.Saver()
        with tf.device('/cpu:0'):
            restore_path = args.checkpoint_path
            train_saver.restore(sess, restore_path)
        return model

    def _prepare_image(self, image):
        interpolation_choice = cv2.INTER_LINEAR \
            if np.product(image.shape)/3 > self._input_width * self._input_height \
            else cv2.INTER_CUBIC
        image = cv2.resize(image, (self._input_height, self._input_width),
                           interpolation=interpolation_choice)
        image = image[:, :, :3].astype(np.float32)
        image[:, :, 0] = (image[:, :, 0] - 123.68) * 0.017  # Normalizes image
        image[:, :, 1] = (image[:, :, 1] - 116.78) * 0.017
        image[:, :, 2] = (image[:, :, 2] - 103.94) * 0.017
        return image

    def _upscale_depth(self, depth_image, original_width, original_height):
        return cv2.resize(depth_image, (original_width, original_height),
                          interpolation=cv2.INTER_CUBIC)

    def predict(self, rgb_image):
        input_image = self._prepare_image(rgb_image)
        if self._model:
            input_images = np.expand_dims(input_image, axis=0)
            
            with graph.as_default():
                depth_result = sess.run(
                    [self._model.depth_est],
                    feed_dict={self._image_tensor: input_images})
            if self._do_upscaling:
                return self._upscale_depth(depth_result[0].squeeze(),
                                           rgb_image.shape[1],
                                           rgb_image.shape[0])
            else:
                return depth_result[0].squeeze()


if __name__ == '__main__':
    # Argument Parser
    args = build_parser().parse_args()

    model_dir = os.path.join("./models", args.model_name)
    sys.path.append(model_dir)
    for key, val in vars(__import__(args.model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val

    bts_predictor = BTSDepthPredictor(
        encoder=args.encoder,
        max_depth=args.max_depth,
        focal_length_px=tf.constant([518.8579]),
        upscaling=args.upscaled)

    dir_to_process = pathlib.Path(args.images_folder).resolve()
    if args.recursive_search:
        images_to_process = glob(str(dir_to_process/"**/*.jpg"), recursive=True)
    else:
        images_to_process = glob(str(dir_to_process/"*.jpg"))
    images_to_process.sort()
    print("{count} images to process.".format(count=len(images_to_process)))

    # Folder structure (RECURSIVELY)
    # For a given path: /some/folder/with/images/subfolder/s/0000011102.jpg
    # create folder structure:
    # /some/folder/with/
    #                L stereo
    #                  L depth_maps
    #                    L subfolder/s
    #                      L 0000011102.jpg.nn.bin
    created_dirs = list()
    suffix_depth_maps = '.nn.bin'
    for image_location in images_to_process:
        image_rgb = cv2.cvtColor(cv2.imread(image_location), cv2.COLOR_BGR2RGB)
        raw_depth_result = bts_predictor.predict(image_rgb)

        if args.save_depth_maps:
            image_path = pathlib.Path(image_location)
            depth_map_dir = None

            if args.recursive_search:
                all_subfolders = list(image_path.parts[:-1])
                all_subfolders.reverse()
                dir_to_process_parts = list(dir_to_process.parent.parts)
                dir_to_process_parts.reverse()
                for level in range(len(dir_to_process_parts)+1):
                    all_subfolders.pop()
                all_subfolders.append('depth_maps')
                all_subfolders.append('stereo')
                all_subfolders.reverse()
                depth_map_dir = dir_to_process.parent / '/'.join(all_subfolders)
            else:
                subfolders_from_filename = 'stereo/depth_maps/' + '/'.join(image_path.name.split('_')[:-1])
                depth_map_dir = image_path.parent/subfolders_from_filename

            depth_map_filename = image_path.name + suffix_depth_maps
            if str(depth_map_dir) not in created_dirs:
                os.makedirs(str(depth_map_dir), exist_ok=True)
                created_dirs.append(str(depth_map_dir))

            write_array(raw_depth_result, str(depth_map_dir/depth_map_filename))
    print('Done! All images processed successfully.')
