# -- encoding:utf-8 --
"""
Created by MengCheng Ren on 2019/7/18
"""
import os
import config as cfg
import random
import numpy as np
import cv2
import tensorflow as tf
import argparse


class DataSet(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.nbr_train_1w = 0
        # self.nbr_train_b = 0

        self.train_data = './train_data/train.csv'

    def train_1w_loader(self):
        print("train data loading...")
        data_list = open(self.train_data, 'r').readlines()
        self.nbr_train_1w = len(data_list)
        batch_size = self.batch_size
        img_width = cfg.IMG_WIDTH
        img_height = cfg.IMG_HEIGHT
        shuffle = True
        nbr_classes = cfg.NUM_CLASS
        grid_attr = cfg.GRID_ATTR

        self.num_data = len(data_list)
        return self.generator_batch(data_list, path='train_1w', grid_attr=grid_attr, nbr_classes=nbr_classes,
                                    batch_size=batch_size, img_height=img_height,
                                    img_width=img_width, shuffle=shuffle)

    def generator_batch(self, data_list, path, grid_attr, nbr_classes, batch_size=1,
                        img_width=416, img_height=416, shuffle=True):
        # 训练集的数量
        N = self.num_data

        new_data_list = []
        new_data_list_append = new_data_list.append
        # 去掉label中的空格和多余的符号
        for item in data_list:
            path_name, tag = item.strip().split(',')
            img_path = os.path.join(os.getcwd(), 'train_data', 'image', path_name)
            if os.path.isfile(img_path):
                new_data_list_append(f"{path_name},{tag}")
        data_list = new_data_list

        # 每个gride_cell的维度
        total_grid_attr = grid_attr + nbr_classes

        if shuffle:
            random.shuffle(data_list)

        # batch_index的计数器
        batch_index = 0
        # 开始生成batch_size的数据
        while True:
            # 每个batch的开始index
            current_index = (batch_index * batch_size) % N
            # 如果没有读到数据集的最后一个batch就继续按照batch_size读取
            if N >= (current_index + batch_size):
                current_batch_size = batch_size
                batch_index += 1
            # 读取到数据集的最后一个batch，batch_size的大小就是最后所有数据的大小， 读完后进行shuffle
            else:
                current_batch_size = N - current_index
                batch_index = 0
                if shuffle:
                    random.shuffle(data_list)

            # 存储图像数据
            X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
            # scale0保存第一个scale的label， scale1和scale2以此类推，每个scale的大小不一样，详情看原论文
            Y_batch_scal0 = np.zeros((current_batch_size, cfg.SCALES[0], cfg.SCALES[0], total_grid_attr))
            Y_batch_scal1 = np.zeros((current_batch_size, cfg.SCALES[1], cfg.SCALES[1], total_grid_attr))
            Y_batch_scal2 = np.zeros((current_batch_size, cfg.SCALES[2], cfg.SCALES[2], total_grid_attr))

            # 开始构建每一个batch_size的数据
            for i in range(current_index, current_index + current_batch_size):
                file = data_list[i].strip().split(',')

                # print(file[0])
                # 将每个数据中object的位置变成一个列表
                position_list = list(map(lambda x: x.split('_'), file[1].split(';')))
                # 图像文件的路径
                path = os.path.join('./train_data', 'image', file[0])
                # 如果该图像不存在就继续下一个数据
                if not os.path.isfile(path):
                    continue
                # 读取图像
                img = cv2.imread(path)
                img = cv2.bilateralFilter(img, 0, 10, 5)
                img = img / 255
                height, width, _ = img.shape
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # 改变图像大小为网络的输入大小
                img = cv2.resize(img, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), interpolation=cv2.INTER_AREA)

                # 存放三种scale的label
                scales_labels = []

                for scale_size in cfg.SCALES:
                    maxtric = np.zeros((scale_size, scale_size, total_grid_attr), np.float32)

                    for position in position_list:
                        if position[0] == '' or len(position) != 5:
                            continue

                        # 将标注的坐标和宽高对应到输入大小的图像上
                        w = float(position[2]) * (cfg.IMG_WIDTH / width)
                        h = float(position[3]) * (cfg.IMG_HEIGHT / height)
                        # x, y是中心点坐标
                        x = (float(position[0]) * (cfg.IMG_WIDTH / width) + w / 2) / (416 / scale_size)
                        y = (float(position[1]) * (cfg.IMG_HEIGHT / height) + h / 2) / (416 / scale_size)
                        # 类别没有坐标，所以不要进行坐标转换
                        c = float(position[4])

                        grid_x = int(x)
                        grid_y = int(y)

                        # 构建label，分别为置信度，中心点左边，宽高，和类别
                        maxtric[grid_y, grid_x, 0] = 1.0
                        maxtric[grid_y, grid_x, 1] = x
                        maxtric[grid_y, grid_x, 2] = y
                        maxtric[grid_y, grid_x, 3] = w
                        maxtric[grid_y, grid_x, 4] = h
                        maxtric[grid_y, grid_x, 5] = c

                    scales_labels.append(maxtric)

                X_batch[i - current_index] = img
                Y_batch_scal0[i - current_index] = scales_labels[0]
                Y_batch_scal1[i - current_index] = scales_labels[1]
                Y_batch_scal2[i - current_index] = scales_labels[2]

            X_batch = X_batch.astype(np.float32)
            Y_batch = [Y_batch_scal0, Y_batch_scal1, Y_batch_scal2]

            yield (X_batch, Y_batch)


class Tensors(object):
    def __init__(self, args):
        total_grid_cell_attr = 5 + cfg.NUM_CLASS

        # DarkNet53参数配置
        self.num_class = cfg.NUM_CLASS
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.scale = self.image_size / self.cell_size
        self.num_anchors = cfg.NUM_ANCHORS
        self.anchors = cfg.ANCHORS
        self.anchor_mask = cfg.ANCHOR_MASK
        self.drop_rate = cfg.DROP_RATE

        self.ignore_thresh = cfg.IGNORE_THRESH

        # _Leaky_Relu config
        self.alpha = cfg.ALPHA

        self.training = args.is_training
        self.batch_size = args.batch_size

        # 输入
        self.inputs_x = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3])
        # 输出
        self.inputs_y = [tf.placeholder(tf.float32, [None, cfg.SCALES[0], cfg.SCALES[0], total_grid_cell_attr]),
                         tf.placeholder(tf.float32, [None, cfg.SCALES[1], cfg.SCALES[1], total_grid_cell_attr]),
                         tf.placeholder(tf.float32, [None, cfg.SCALES[2], cfg.SCALES[2], total_grid_cell_attr])]

        # self.detection_layer(self.training)

        self.loss = 0

        # loss config
        self.object_alpha = cfg.OBJECT_ALPHA
        self.no_object_alpha = cfg.NO_OBJECT_ALPHA
        self.class_alpha = cfg.CLASS_ALPHA
        self.coord_alpha = cfg.COORD_ALPHA

        if self.training:
            self.total_loss(self.inputs_y)
        else:
            self.detection_layer(args.is_training)

    def net(self, inputs, is_training):
        with tf.variable_scope('visualization'):
            tf.summary.image("image", inputs)

        with tf.variable_scope('darknet53'):
            num_layer = 0

            # 0
            layer = self.Conv2d(inputs, filters=32, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                training=is_training, name='_Conv2d_' + str(num_layer))
            num_layer += 1

            # 1
            layer = self.Conv2d(layer, filters=64, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                training=is_training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            # 2 - 4
            for _ in range(1):
                layer = self.Conv2d(layer, filters=32, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Conv2d(layer, filters=64, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Residual(layer, shortcut)

            # 5
            layer = self.Conv2d(layer, filters=128, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                training=is_training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            # 6 - 9
            for _ in range(2):
                layer = self.Conv2d(layer, filters=64, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Conv2d(layer, filters=128, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Residual(layer, shortcut)

            # 10
            layer = self.Conv2d(layer, filters=256, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                training=is_training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            with tf.variable_scope('visualization'):
                layer10_image = layer[0:1, :, :, 0:256]
                layer10_image = tf.transpose(layer10_image, perm=[3, 1, 2, 0])
                tf.summary.image('layer45_image', layer10_image, max_outputs=255)

            # 11 - 26
            for _ in range(8):
                layer = self.Conv2d(layer, filters=128, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Conv2d(layer, filters=256, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Residual(layer, shortcut)
            self.scale_2 = layer

            # 27
            layer = self.Conv2d(layer, filters=512, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                training=is_training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            with tf.variable_scope('visualization'):
                layer27_image = layer[0:1, :, :, 0:512]
                layer27_image = tf.transpose(layer27_image, perm=[3, 1, 2, 0])
                tf.summary.image('layer45_image', layer27_image, max_outputs=255)

            # 28 - 44
            for _ in range(8):
                layer = self.Conv2d(layer, filters=256, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Conv2d(layer, filters=512, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Residual(layer, shortcut)
            self.scale_1 = layer

            # 45
            layer = self.Conv2d(layer, filters=1024, shape=[3, 3], stride=(2, 2), alpha=self.alpha,
                                training=is_training, name='_Conv2d_' + str(num_layer))
            num_layer += 1
            shortcut = layer

            with tf.variable_scope('visualization'):
                layer45_image = layer[0:1, :, :, 0:1024]
                layer45_image = tf.transpose(layer45_image, perm=[3, 1, 2, 0])
                tf.summary.image('layer45_image', layer45_image, max_outputs=255)

            # 46 - 53
            for _ in range(4):
                layer = self.Conv2d(layer, filters=512, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Conv2d(layer, filters=1024, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                    training=is_training, name='_Conv2d_' + str(num_layer))
                num_layer += 1
                layer = self.Residual(layer, shortcut)
            self.scale_0 = layer

    def detection_layer(self, training=True):
        self.net(self.inputs_x, self.training)
        with tf.name_scope('detection_layer'):
            with tf.variable_scope('scale_0'):
                # with tf.device('/gpu:0'):
                self.scale_0 = self.Conv2d(self.scale_0, filters=512, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_1')
                self.scale_0 = self.Conv2d(self.scale_0, filters=1024, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_2')
                self.scale_0 = self.Conv2d(self.scale_0, filters=512, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_3')

                with tf.variable_scope('visualization'):
                    layer10_image = self.scale_2[0:1, :, :, 0:256]
                    layer10_image = tf.transpose(layer10_image, perm=[3, 1, 2, 0])
                    tf.summary.image('layer45_image', layer10_image, max_outputs=255)

                self.scale_0 = self.Conv2d(self.scale_0, filters=1024, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_4')
                self.scale_0 = self.Conv2d(self.scale_0, filters=512, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_5')
                layer_final = self.UpSampling2d(self.scale_0, 256, shape=[1, 1], strides=(2, 2),
                                                name='_UpSampling2d')
                self.scale_0 = self.Conv2d(self.scale_0, filters=1024, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_6')
                self.scale_0 = self.Conv2d(self.scale_0,
                                           filters=(self.num_class + 5) * self.num_anchors,
                                           shape=[1, 1],
                                           stride=(1, 1),
                                           alpha=self.alpha,
                                           training=training,
                                           name='_Conv2d_output')

            with tf.variable_scope('scale_1'):
                # with tf.device('/gpu:0'):
                self.scale_1 = self.Conv2d(self.scale_1, filters=256, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_1')
                self.scale_1 = tf.concat([self.scale_1, layer_final], 3, name='concat_scale_0_to_scale_1')
                self.scale_1 = self.Conv2d(self.scale_1, filters=512, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_2')
                self.scale_1 = self.Conv2d(self.scale_1, filters=256, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_3')

                with tf.variable_scope('visualization'):
                    layer10_image = self.scale_2[0:1, :, :, 0:256]
                    layer10_image = tf.transpose(layer10_image, perm=[3, 1, 2, 0])
                    tf.summary.image('layer45_image', layer10_image, max_outputs=255)

                self.scale_1 = self.Conv2d(self.scale_1, filters=512, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_4')
                self.scale_1 = self.Conv2d(self.scale_1, filters=256, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_5')
                layer_final = self.UpSampling2d(self.scale_1, 128, shape=[1, 1], strides=(2, 2),
                                                name='_UpSampling2d')
                self.scale_1 = self.Conv2d(self.scale_1, filters=512, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_6')
                self.scale_1 = self.Conv2d(self.scale_1,
                                           filters=(self.num_class + 5) * self.num_anchors,
                                           shape=[1, 1],
                                           stride=(1, 1),
                                           alpha=self.alpha,
                                           training=training,
                                           name='_Conv2d_output')

            with tf.variable_scope('scale_2'):
                # with tf.device('/gpu:0'):
                self.scale_2 = self.Conv2d(self.scale_2, filters=128, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_1')
                self.scale_2 = tf.concat([self.scale_2, layer_final], 3,
                                         name='concat_scale_1_to_scale_2')
                self.scale_2 = self.Conv2d(self.scale_2, filters=256, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_2')
                self.scale_2 = self.Conv2d(self.scale_2, filters=128, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_3')

                with tf.variable_scope('visualization'):
                    layer10_image = self.scale_2[0:1, :, :, 0:256]
                    layer10_image = tf.transpose(layer10_image, perm=[3, 1, 2, 0])
                    tf.summary.image('layer45_image', layer10_image, max_outputs=255)

                self.scale_2 = self.Conv2d(self.scale_2, filters=256, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_4')
                self.scale_2 = self.Conv2d(self.scale_2, filters=128, shape=[1, 1], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_5')
                self.scale_2 = self.Conv2d(self.scale_2, filters=256, shape=[3, 3], stride=(1, 1), alpha=self.alpha,
                                           training=training, name='_Conv2d_6')
                self.scale_2 = self.Conv2d(self.scale_2,
                                           filters=(self.num_class + 5) * self.num_anchors,
                                           shape=[1, 1],
                                           stride=(1, 1),
                                           alpha=self.alpha,
                                           training=training,
                                           name='_Conv2d_output')
        self.scales = [self.scale_0, self.scale_1, self.scale_2]

    def total_loss(self, labels, scope='total_loss'):
        self.detection_layer(self.training)
        with tf.name_scope(scope):
            grid_shape = [tf.cast(tf.shape(self.scales[i])[1:3], dtype=tf.float32) for i in range(len(self.scales))]

            for i in range(len(self.scales)):
                grid, predict, bbox_xy, bbox_wh = self.scale_hat(self.scales[i],
                                                                 self.anchors[self.anchor_mask[i]],
                                                                 calculate_loss=True)
                pred_bbox = tf.concat([bbox_xy, bbox_wh], axis=-1)

                # label has object grid
                label_object_mask = tf.cast(labels[i][..., 0:1], tf.float32)
                label_object_mask = tf.expand_dims(label_object_mask, axis=3)
                label_object_mask = tf.tile(label_object_mask, [1, 1, 1, self.num_anchors, 1])
                # label no object grid
                label_no_object_mask = 1.0 - label_object_mask
                # label xy
                label_xy = labels[i][..., 1:3]
                label_xy = tf.expand_dims(label_xy, axis=3) / tf.cast(grid_shape[i], tf.float32)
                label_xy = tf.tile(label_xy, [1, 1, 1, self.num_anchors, 1])
                # label wh
                label_wh = labels[i][..., 3:5] / tf.cast(self.image_size, tf.float32)
                label_wh = tf.expand_dims(label_wh, axis=3)
                label_wh = tf.tile(label_wh, [1, 1, 1, self.num_anchors, 1])
                # label class
                label_class = tf.cast(labels[i][..., 5:], tf.float32)
                label_class = tf.expand_dims(label_class, axis=3)
                label_class = tf.tile(label_class, [1, 1, 1, self.num_anchors, 1])

                label_box = tf.concat([label_xy, label_wh], axis=-1)

                # pred confidence
                confidence_hat = tf.sigmoid(predict[..., 0:1])    # (None, 13, 13, 5, 1)
                # pred xywh
                txy_hat = tf.sigmoid(predict[..., 1:3])
                # txy_hat = predict[..., 1:3]
                tf.summary.histogram('txy_hat', txy_hat)
                twh_hat = predict[..., 3:5]
                tf.summary.histogram('twh_hat', twh_hat)
                # pred class
                class_hat = predict[..., 5:]

                # ground truth xywh
                anchors_tensor = tf.cast(self.anchors[self.anchor_mask[i]], tf.float32)
                anchors_tensor = tf.reshape(anchors_tensor, [1, 1, self.num_anchors, 2])
                # twh = tf.log(labels[i][..., 3:5] / anchors_tensor)
                twh = tf.log(label_wh * tf.cast(self.image_size, tf.float32) / anchors_tensor)
                # avoid log(0)
                twh = tf.keras.backend.switch(label_object_mask, twh, tf.zeros_like(twh))
                txy = (label_xy * tf.cast(grid_shape[i], tf.float32) - grid) * label_object_mask

                # iou
                iou = self.calculate_IOU(pred_bbox, label_box)  # (None, 13, 13, 3)
                # get best confidence for each grid cell
                best_confidence = tf.reduce_max(confidence_hat, axis=-1, keepdims=True)
                best_confidence_mask = tf.cast(confidence_hat >= best_confidence, tf.float32)
                self.best_confidence_mask_test = best_confidence_mask
                # get ignore mask, if some anchor box of the object grid cell hasn't best iou but they may be has better iou with ground truth
                ignore_mask = tf.cast(iou < self.ignore_thresh, tf.float32)
                ignore_mask = tf.expand_dims(ignore_mask, axis=4)

                label_object_confidence = 1
                label_no_object_confidence = 0

                # if not the best iou box, calculate no_object_mask
                no_object_mask = (1 - best_confidence_mask) + label_no_object_mask

                object_confidence_loss = self.calculate_object_confidence_loss(confidence_hat,
                                                                               label_object_confidence,
                                                                               label_object_mask,
                                                                               best_confidence_mask,
                                                                               'object_confidence_loss_' + str(i))
                no_object_confidence_loss = self.calculate_no_object_confidence_loss(confidence_hat,
                                                                                     label_no_object_confidence,
                                                                                     no_object_mask,
                                                                                     ignore_mask,
                                                                                     'no_object_confidence_loss_' + str(i))
                xy_loss = self.calculate_xy_loss(label_object_mask, best_confidence_mask, txy_hat, txy, 'xy_loss_' + str(i))
                wh_loss = self.calculate_wh_loss(label_object_mask, best_confidence_mask, twh_hat, twh, 'wh_loss_' + str(i))
                class_loss = self.calculate_classify_loss(label_object_mask, class_hat, label_class, 'classify_loss_' + str(i))

                self.loss += self.coord_alpha * (xy_loss + wh_loss) + \
                             self.object_alpha * object_confidence_loss + \
                             self.no_object_alpha * no_object_confidence_loss + \
                             self.class_alpha * class_loss

                tmp = tf.expand_dims(iou, axis=4)
                avg_iou = tf.reduce_mean(tf.boolean_mask(tmp, tf.cast(label_object_mask, tf.bool)))

                tf.summary.scalar('avg_iou' + str(i), avg_iou)
                tf.summary.scalar('object_confidence_loss' + str(i), object_confidence_loss)
                tf.summary.scalar('no_object_confidence_loss' + str(i), no_object_confidence_loss)
                tf.summary.scalar('xy_loss' + str(i), xy_loss)
                tf.summary.scalar('wh_loss' + str(i), wh_loss)
                tf.summary.scalar('class_loss' + str(i), class_loss)
            tf.summary.scalar('loss', self.loss)

    def calculate_IOU(self, pred_box, label_box, scope='IOU'):
        with tf.name_scope(scope):
            with tf.name_scope('pred_box'):
                pred_box_xy = pred_box[..., 0:2]
                pred_box_wh = pred_box[..., 2:]
                pred_box_wh_half = pred_box_wh / 2.0
                pred_box_leftup = pred_box_xy - pred_box_wh_half
                pred_box_rightdown = pred_box_xy + pred_box_wh_half
                pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]

            with tf.name_scope('label_box'):
                label_box_xy = label_box[..., 0:2]
                label_box_wh = label_box[..., 2:]
                label_box_wh_half = label_box_wh / 2.0
                label_box_leftup = label_box_xy - label_box_wh_half
                label_box_rightdown = label_box_xy + label_box_wh_half
                label_box_area = label_box_wh[..., 0] * label_box_wh[..., 1]

            with tf.name_scope('intersection'):
                intersection_leftup = tf.maximum(pred_box_leftup, label_box_leftup)
                intersection_rightdown = tf.minimum(pred_box_rightdown, label_box_rightdown)
                intersection_wh = tf.maximum(intersection_rightdown - intersection_leftup, 0.)
                intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]

            iou = tf.divide(intersection_area, (pred_box_area + label_box_area - intersection_area), name='IOU_result')
        return iou

    def scale_hat(self, feature_map, anchors, calculate_loss=False):
        anchors_tensor = tf.cast(anchors, tf.float32)
        anchors_tensor = tf.reshape(anchors_tensor, [1, 1, 1, self.num_anchors, 2])

        with tf.name_scope('create_grid_offset'):
            grid_shape = tf.shape(feature_map)[1:3]
            grid_y = tf.range(0, grid_shape[0])
            grid_x = tf.range(0, grid_shape[1])
            grid_y = tf.reshape(grid_y, [-1, 1, 1, 1])  # (13, 1, 1, 1)
            grid_x = tf.reshape(grid_x, [1, -1, 1, 1])  # (1, 13, 1, 1)
            grid_y = tf.tile(grid_y, [1, grid_shape[0], 1, 1])  # (13, 13, 1, 1)
            grid_x = tf.tile(grid_x, [grid_shape[1], 1, 1, 1])  # (13, 13, 1, 1)
            grid = tf.concat([grid_x, grid_y], axis=3)
            grid = tf.cast(grid, tf.float32)

        feature_map = tf.reshape(feature_map, [-1, grid_shape[0], grid_shape[1], self.num_anchors, 5 + self.num_class])

        with tf.name_scope('scale_hat_activations'):
            tf.summary.histogram('feature_map', feature_map)

            bbox_confidence = tf.sigmoid(feature_map[..., 0:1], name='confidence')
            tf.summary.histogram('confidence', bbox_confidence)
            bbox_xy = tf.sigmoid(feature_map[..., 1:3], name='xy')
            tf.summary.histogram('xy', bbox_xy)
            bbox_wh = tf.exp(feature_map[..., 3:5], name='wh')
            #tf.summary.histogram('wh', bbox_wh)
            bbox_class_probs = tf.sigmoid(feature_map[..., 5:], name='class_probs')
            tf.summary.histogram('class_probs', bbox_class_probs)

            bbox_xy = (bbox_xy + grid) / tf.cast(grid_shape[0], tf.float32)
            bbox_wh = bbox_wh * anchors_tensor / tf.cast(self.image_size, tf.float32)

        if calculate_loss:
            return grid, feature_map, bbox_xy, bbox_wh

        return bbox_xy, bbox_wh, bbox_confidence, bbox_class_probs

    def calculate_object_confidence_loss(self, object_confidence_hat, object_confidence, object_mask,
                                         best_confidence_mask, scope='object_confidence_loss'):
        with tf.name_scope(scope):
            object_loss = tf.reduce_sum(object_mask * tf.square(object_confidence - object_confidence_hat) *
                                        best_confidence_mask) / self.batch_size

        return object_loss

    def calculate_no_object_confidence_loss(self, no_object_confidence_hat, no_object_confidence,
                                            no_object_mask, ignore_mask, scope='no_object_confidence_loss'):
        with tf.name_scope(scope):
            no_object_loss = tf.reduce_sum(no_object_mask * tf.square(no_object_confidence - no_object_confidence_hat) *
                                           ignore_mask) / self.batch_size
        return no_object_loss

    def calculate_xy_loss(self, label_object_mask, best_confidence_mask, txy_hat, txy, scope='xy_loss'):
        with tf.name_scope(scope):
            xy_loss = tf.reduce_sum(label_object_mask * best_confidence_mask *
                                    tf.square(txy - txy_hat)) / self.batch_size

        return xy_loss

    def calculate_wh_loss(self, label_object_mask, best_confidence_mask, twh_hat, twh, scope='wh_loss'):
        with tf.name_scope(scope):
            wh_loss = tf.reduce_sum(label_object_mask * best_confidence_mask * tf.square(twh - twh_hat)) / \
                      self.batch_size

        return wh_loss

    def calculate_classify_loss(self, object_mask, predicts_class, labels_class, scope='classify_loss'):
        with tf.name_scope(scope):
            class_loss = tf.reduce_sum(object_mask *
                                       tf.keras.backend.binary_crossentropy(labels_class,
                                                                            predicts_class,
                                                                            from_logits=True)) / self.batch_size
        return class_loss

    def Conv2d(self, inputs, filters, shape, stride=(1, 1),
               alpha=0.01, is_drop_out=False, is_batch_normal=True, is_Leaky_Relu=True,
               training=True, name=None):
        layer = tf.layers.conv2d(inputs,
                                 filters,
                                 shape,
                                 stride,
                                 padding='SAME',
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02, dtype=tf.float32),
                                 name=name)
        if is_drop_out:
            layer = tf.layers.dropout(layer, self.drop_rate, training=training)

        if is_batch_normal:
            layer = tf.layers.batch_normalization(layer, training=training)

        if is_Leaky_Relu:
            layer = self.Leaky_Relu(layer, alpha)
        return layer

    def Residual(self, conv, shortcut, alpha=0.01):
        res = self.Leaky_Relu(conv + shortcut, alpha)
        return res

    def Leaky_Relu(self, input, alpha=0.01):
        output = tf.maximum(input, tf.multiply(input, alpha))
        return output

    def UpSampling2d(self, inputs, filters, shape=(1, 1), strides=(2, 2), name=None):
        layer = tf.layers.conv2d_transpose(inputs, filters, shape, strides, name=name)
        return layer

    def get_box_and_score(self, scale, anchors, x_scale, y_scale):
        bbox_xy, bbox_wh, bbox_confidence, bbox_class_probs = self.scale_hat(scale, anchors)

        score = bbox_confidence * bbox_class_probs
        score = tf.reshape(score, [-1, self.num_class])

        scale_size = scale.get_shape().as_list()[1]
        bbox_xy = bbox_xy * scale_size * (self.image_size / scale_size) / tf.cast([x_scale, y_scale],
                                                                                  tf.float32)
        bbox_xy = tf.reshape(bbox_xy, [-1, 2])
        bbox_yx = bbox_xy[..., ::-1]

        bbox_wh = bbox_wh * self.image_size / tf.cast([x_scale, y_scale], tf.float32)
        bbox_wh = tf.reshape(bbox_wh, [-1, 2])
        bbox_hw = bbox_wh[..., ::-1]

        bbox_y1x1 = bbox_yx - (bbox_hw / 2.0)
        bbox_y2x2 = bbox_yx + (bbox_hw / 2.0)

        box = tf.concat([bbox_y1x1, bbox_y2x2], axis=-1)

        return box, score

    def predict(self, score_threshold, iou_threshold, max_boxes, x_scale, y_scale):
        boxes = []
        boxes_score = []

        for i in range(len(self.scales)):
            with tf.name_scope('predict' + str(i)):
                box, score = self.get_box_and_score(self.scales[i], cfg.ANCHORS[cfg.ANCHOR_MASK[i]], x_scale, y_scale)

                boxes.append(box)
                boxes_score.append(score)

        boxes = tf.concat(boxes, axis=0)

        boxes_score = tf.concat(boxes_score, axis=0)

        mask = boxes_score >= score_threshold
        max_boxes = tf.constant(max_boxes, dtype='int32', name='max_boxes')

        result_boxes = []
        result_score = []
        result_classes = []
        for num in range(self.num_class):
            class_boxes = tf.boolean_mask(boxes, mask[:, num])

            class_boxes_score = tf.boolean_mask(boxes_score[:, num], mask[:, num])

            nms_index = tf.image.non_max_suppression(class_boxes,
                                                     class_boxes_score,
                                                     max_boxes,
                                                     iou_threshold=iou_threshold,
                                                     name='NMS')
            class_boxes = tf.gather(class_boxes, nms_index)
            class_boxes_score = tf.gather(class_boxes_score, nms_index)
            classes = tf.ones_like(class_boxes_score, 'int32')

            result_boxes.append(class_boxes)
            result_score.append(class_boxes_score)
            result_classes.append(classes)

        result_boxes = tf.concat(result_boxes, axis=0)
        result_score = tf.concat(result_score, axis=0)
        result_classes = tf.concat(result_classes, axis=0)

        return result_boxes, result_score, result_classes


class YOLOv3(object):
    def __init__(self, args):
        # 学习率参数设置
        self.args = args
        base_learning_rate = args.learning_rate
        decay_steps = args.decay_steps
        decay_rate = args.decay_rate

        # 迭代次数
        self.epochs = args.epochs

        # 构建默认图
        graph = tf.Graph()
        with graph.as_default():
            self.net = Tensors(args)

            # 将之前的summary统一化管理
            self.summary_op = tf.summary.merge_all()
            # 构建summary文件写入器
            self.writer = tf.summary.FileWriter(cfg.SUMMARY_PATH)
            # 保存模型类
            self.saver = tf.train.Saver()

            if args.is_training:
                # 学习率设置
                global_step = tf.train.create_global_step()
                learning_rate = tf.train.exponential_decay(base_learning_rate, global_step,
                                                           decay_steps, decay_rate, staircase=False)
                # 构建优化器
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = optimizer.minimize(self.net.loss)

            # 配置gpu
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

            if args.is_training:
                try:
                    self.sess.run(tf.global_variables_initializer())
                    self.saver.restore(self.sess, cfg.MODEL_PATH)
                    print("restoring model successful...")
                except:
                    self.sess.run(tf.global_variables_initializer())
                    print("restoring model failed...")
            else:
                try:
                    self.saver.restore(self.sess, cfg.MODEL_PATH)
                    print("restoring model successful...")
                except:
                    print("model not exist...")

    def train(self):
        data = DataSet(args)
        train_batch = data.train_1w_loader()

        for epoch in range(self.epochs):
            for step in range(data.num_data // args.batch_size):
                batch = next(train_batch)
                feed_dict = {self.net.inputs_x: batch[0],
                             self.net.inputs_y[0]: batch[1][0],
                             self.net.inputs_y[1]: batch[1][1],
                             self.net.inputs_y[2]: batch[1][2]}
                _, total_loss, summary = self.sess.run([self.train_op, self.net.loss, self.summary_op],
                                                       feed_dict=feed_dict)
                print("Epoch:{}, step:{}, loss:{}".format(epoch, step, total_loss))

                if step % 100 == 0:
                    self.writer.add_summary(summary)
                    self.saver.save(self.sess, cfg.MODEL_PATH)

    def detect(self, image):
        height, width, _ = image.shape
        x_scale = cfg.IMG_WIDTH / width
        y_scale = cfg.IMG_HEIGHT / height

        score_threshold = args.score_threshold
        iou_threshold = args.iou_threshold
        max_boxes = args.max_boxes

        box, score, classes = self.net.predict(score_threshold, iou_threshold, max_boxes, x_scale, y_scale)

        img = cv2.resize(image, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0)
        # self.anchor_mask = cfg.ANCHOR_MASK
        # self.anchors = cfg.ANCHORS
        print(img.shape)
        print(img.shape)
        # img = np.reshape(img, (1, cfg.IMG_HEIGHT, cfg.IMG_HEIGHT, 3))
        result_boxes, result_score, result_classes = self.sess.run([box, score, classes],
                                                                   feed_dict={self.net.inputs_x: img})

        self.draw(img, result_boxes)

    def draw(self, image, boxes):
        for box in boxes:
            x, y, w, h = box

            if w * h <= 200.0:
                continue

            x1 = max(0, np.floor(x + 0.5).astype(int))
            y1 = max(0, np.floor(y + 0.5).astype(int))
            x2 = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
            y2 = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imwrite('./result/test.jpg', image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_training", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('--decay_rate', type=float, default=0.7)
    parser.add_argument('--decay_steps', type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    # NMS config
    parser.add_argument('--max_boxes', default=35, type=int)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--score_threshold', default=0.3, type=float)
    parser.add_argument("--data_path", default='./test_data/1.jpg', type=str)
    args = parser.parse_args()

    image = cv2.imread(args.data_path)
    tensor = YOLOv3(args)
    if args.is_training:
        tensor.train()
    else:
        tensor.detect(image)
