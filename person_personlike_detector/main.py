import ultralytics
from ultralytics import YOLO
# import Ipython
# from Ipython.display import Image
import os
import glob
import numpy as np
import pandas as pd
import cv2


ultralytics.checks()

#DASHBOARD

import streamlit as st
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

path = 'C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/transfer_learning.pt'
model_ = YOLO(path)
model_o = YOLO("yolo11s.pt")

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.from_numpy(img).type(torch.uint8) #torch.tensor(img)
    result = prediction[0]
    boxes = result.boxes.xyxy
    class_ids = result.boxes.cls
    labels = [result.names[int(cls_id)]for cls_id in class_ids]
    colors = ["red" if int(cls_id)== 0 else "green" for cls_id in class_ids]
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes = boxes, labels = labels, 
                                          colors =colors, width = 2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0)
    return img_with_bboxes_np

st.title("Pedestrian and Person Like Object Detector")
st.sidebar.header('Model Configurations')
objective = st.sidebar.radio("Task", ["Original YOLO", "Transfer Learned YOLO"])
confidence_value = float(st.sidebar.slider("Select Model Confidence", 25,100,40))/100 

upload = st.file_uploader(label="Upload Image Here:", type = ["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload)
    # model = model_o if objective == "Original YOLO" else model_
    prediction = model_(img)
    img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction)   # giving invalid shape (3, 500, 353)for image data error

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    plt.imshow(img_with_bbox)
    plt.xticks([],[])
    plt.yticks([],[])
    ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

    st.pyplot(fig, use_container_width = True)

    del prediction[0].boxes
    st.header("Predicted Probabilities")
    st.write(prediction)
