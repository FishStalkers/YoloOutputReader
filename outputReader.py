import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


def __construct_box(x_center, y_center, width, height):
    x_left = x_center - width / 2
    x_right = x_center + width / 2
    y_top = y_center - height / 2
    y_bottom = y_center + height / 2

    return x_left, y_top, x_right, y_bottom


def calc_iou(box1, box2):
    x1, y1, x2, y2 = __construct_box(box1[1], box1[2], box1[3], box1[4])
    x3, y3, x4, y4 = __construct_box(box2[1], box2[2], box2[3], box2[4])
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
    return iou


data_manual = "manual_annotation.txt"
data_yolo = "yolo_output.txt"
df_yolo = pd.read_csv(data_yolo, sep=" ", header=None)
df_manual = pd.read_csv(data_manual, sep=" ", header=None)

df_manual.columns = ["category", "x_center", "y_center", "width", "height"]
df_yolo.columns = ["category", "x_center", "y_center", "width", "height", "conf"]


iou_list = []

for i in df_yolo.index:
    box1 = df_yolo.iloc[i].tolist()
    box2 = df_manual.iloc[i].tolist()
    iou = calc_iou(box1, box2)
    iou_list.append(iou)

df_yolo["IOU"] = iou_list

print(df_yolo)

# b_plot = df_yolo.boxplot(by=[0], column=[3])

# b_plot.plot()
# plot.xlabel("Confidence Level")
# plot.ylabel("Classification")
# plot.title("Confidence Level for Each Object")
# plot.show()
