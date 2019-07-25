# -- encoding:utf-8 --
"""
Created by MengCheng Ren on 2019/7/18
"""
import numpy as np

SCALES = [13, 26, 52]

NUM_CLASS = 1
GRID_ATTR = 5

IMG_WIDTH = 416
IMG_HEIGHT = 416

SCORE_THRESHOLD_LIST = np.linspace(0.7, 0.8, 3, dtype=np.float32)
IOU_THRESHOLD_LIST = np.linspace(0.3, 0.4, 3, dtype=np.float32)
MAX_BOXES = 40

train_val_rate = 0.8

CLASSES = ['car']

IMAGE_SIZE = 416

# CELL_SIZE = np.array([13, 26, 52])
# NUM_ANCHORS = 3
# ANCHORS = np.array([[45.068, 27.935],
#                     [76.403, 43.393],
#                     [88.214, 85.294],
#                     [124.864, 118.636],
#                     [136.438, 52.152],
#                     [163.376, 160.992],
#                     [209.207, 196.178],
#                     [227.513, 88.571],
#                     [294.034, 332.865]])
# ANCHOR_MASK = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

CELL_SIZE = np.array([13, 26, 52])
NUM_ANCHORS = 5
ANCHORS = np.array([[39.07201754,  25.80140428],
                    [65.80069799,  31.56669918],
                    [66.88678829, 146.12497191],
                    [70.89768814,  62.98424832],
                    [99.83233678,  86.10561809],
                    [110.42969977,  38.94441885],
                    [128.08780003, 116.24260609],
                    [153.98412516,  61.44485038],
                    [159.62295651, 149.98699965],
                    [172.29549498, 189.13133234],
                    [221.17919093, 263.46911562],
                    [214.65606772, 175.01036746],
                    [223.47169047,  81.53708667],
                    [318.79080674, 152.33352026],
                    [323.17118454, 365.64912624]])
ANCHOR_MASK = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]

DROP_RATE = 0.2

# leakyrelu alpha
ALPHA = 0.1

IGNORE_THRESH = 0.95

OBJECT_ALPHA = 1.0
NO_OBJECT_ALPHA = 0.5
CLASS_ALPHA = 1.0
COORD_ALPHA = 5.0
ATTENTION_ALPHA = 0.5

SUMMARY_PATH = './summary'
MODEL_PATH = './model/yolo3'
