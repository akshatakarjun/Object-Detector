import ultralytics
from ultralytics import YOLO
import Ipython
from Ipython.display import Image
import os
import glob
import numpy as np
import pandas as pd
import cv2


# ultralytics.checks()

# i_s = r'C:\Users\91776\Desktop\Pedestrian_Detection\trial'
# i_d = r'C:\Users\91776\Desktop\Pedestrian_Detection'

# # using existing trained model's weights to predict on a different dataset that contain persn and person-like objects
# model = YOLO("yolo11s.pt")
# source = r"C:\Users\91776\Desktop\Pedestrian_Detection\trial\image1.jpg"
# output = model(source)
# print(f'data type of output {type(output)}')
# print(f'the output as it is is: {output}')
# print(f'shape of the output {output[0].orig_shape}')
# print(output[0].boxes.shape)
# print(output[0]['names'])
# model.info()

# images = glob.glob(i_s+'/*.*')
# for i,img in enumerate(images, start =1):
#     try:
#         results = model(img, 
#                         # conf = 0.65, 
#                         device = 'cpu',  
#                         # visualize = True,
#                         # classes = 0,  
#                         # project = 'C:/Users/91776/Desktop/Pedestrian_Detection/trial_results_from_weights_withoutCLASSidmentioned',
#                         # save = True
#                         )
#         for result in results:
#             output_filename = os.path.join(i_d, f'result_{i}.jpg')
#             result.save(filename=output_filename)

#     except:
#         print('image not saved')

#DASHBOARD

import streamlit as st
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

# model = YOLO("yolo11s.pt")

# def create_image_with_bboxes(img, prediction):
#     img_tensor = torch.tensor(img)
#     result = prediction[0]
#     boxes = result.boxes.xyxy
#     class_ids = result.boxes.cls
#     labels = [result.names[int(cls_id)]for cls_id in class_ids]
#     colors = ["red" if int(cls_id)== 0 else "green" for cls_id in class_ids]
#     img_with_bboxes = draw_bounding_boxes(img_tensor, boxes = boxes, labels = labels, 
#                                           colors =colors, width = 2)
#     img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1,2,0)
#     return img_with_bboxes_np

# st.title("Pedestrian and Person Like Object Detector")
# upload = st.file_uploader(label="Upload Image Here:", type = ["png", "jpg", "jpeg"])

# if upload:
#     img = Image.open(upload)
#     prediction = model(img)
#     img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2,0,1), prediction)   # giving invalid shape (3, 500, 353)for image data error

#     fig = plt.figure(figsize=(12,12))
#     ax = fig.add_subplot(111)
#     plt.imshow(img_with_bbox)
#     plt.xticks([],[])
#     plt.yticks([],[])
#     ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

#     st.pyplot(fig, use_container_width = True)

#     del prediction[0].boxes
#     st.header("Predicted Probabilities")
#     st.write(prediction)



 
# CONVERTING CUSTOM DATA TO TRAIN



import xml.etree.ElementTree as ET
import yaml
import seaborn as sns
import shutil
import random
import warnings as wr
from ultralytics.utils.plotting import Annotator

wr.filterwarnings("ignore")

train_img_xml = ('C:/Users/91776/Desktop/Pedestrian_Detection/xml/Train/JPEGImages')
train_annot_xml = ('C:/Users/91776/Desktop/Pedestrian_Detection/xml/Train/Annotations')
test_img_xml = ('C:/Users/91776/Desktop/Pedestrian_Detection/xml/Test/JPEGImages')
test_annot_xml = ('C:/Users/91776/Desktop/Pedestrian_Detection/xml/Test/Annotations')
val_img_xml = ('C:/Users/91776/Desktop/Pedestrian_Detection/xml/Val/JPEGImages')
val_annot_xml = ('C:/Users/91776/Desktop/Pedestrian_Detection/xml/Val/Annotations')

train_img_yaml = ('C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/train/images')
train_annot_yaml = ('C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/train/labels')
test_img_yaml = ('C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/test/images')
test_annot_yaml = ('C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/test/labels')
val_img_yaml = ('C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/validation/images')
val_annot_yaml = ('C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/validation/labels')

classes = {'person':1, 'person-like':0}

# Getting the dimensions of the image

def image_size(img_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    return width, height

def creating_yaml_data(Annotation_Path):
    info = {'xmin':[], 'ymin':[], 'xmax':[], 'ymax':[], 'ymax':[], 'name':[], 'label':[], 'width':[], 'height':[]}

    for file in sorted(glob.glob(str(Annotation_Path+'/*.xml*'))):
        data=ET.parse(file)
        for element in data.iter():
            if 'object' == element.tag:
                for attribute in list(element):
                    if 'name' in attribute.tag:
                        name = attribute.text
                        info['label'] += [name]
                        base_filename = os.path.basename(file)
                        img_name = [os.path.splitext(base_filename)[0]] #[file.split('/')[-1][0:-4]]
                        info['name'] += img_name

                        img_size = image_size(f'{Annotation_Path[:-11]}JPEGImages/{img_name[0]}.jpg')
                        info['width'].append(img_size[0])
                        info['height'].append(img_size[1])

                        

                    if 'bndbox' == attribute.tag:
                        for dimension in list(attribute):
                            if 'xmin' == dimension.tag:
                                xmin = int(round(float(dimension.text)))
                                info['xmin'] += [xmin]
                            if 'ymin' == dimension.tag:
                                ymin = int(round(float(dimension.text)))
                                info['ymin'] += [ymin]
                            if 'xmax' == dimension.tag:
                                xmax = int(round(float(dimension.text)))
                                info['xmax'] += [xmax]
                            if 'ymax' == dimension.tag:
                                ymax = int(round(float(dimension.text)))
                                info['ymax'] += [ymax]
    return pd.DataFrame(info)

train_yaml = creating_yaml_data(train_annot_xml)
test_yaml = creating_yaml_data(test_annot_xml)
val_yaml = creating_yaml_data(val_annot_xml)

def convert_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height, class_id):
    x_centre = (xmin + xmax) / 2.0
    y_centre = (ymin + ymax) / 2.0
    x_centre /= img_width
    y_centre /= img_height
    box_width = (xmax - xmin)/img_width
    box_height = (ymax-ymin)/img_height

    return f'{class_id} {x_centre} {y_centre} {box_width} {box_height} \n'

def save_yolo_annot(df, dest_path):
    for name, group in df.groupby('name'):
        with open(os.path.join(dest_path, name+'.txt'), 'w') as f:
            for _, row in group.iterrows():
                yolo_format = convert_to_yolo(row['xmin'], row['ymin'], row['xmax'], row['ymax'],
                                              row['width'], row['height'], classes[row['label']])
                f.write(yolo_format)

os.makedirs(train_annot_yaml, exist_ok=True)
os.makedirs(test_annot_yaml, exist_ok=True)
os.makedirs(val_annot_yaml, exist_ok=True)

save_yolo_annot(train_yaml, train_annot_yaml )
save_yolo_annot(test_yaml, test_annot_yaml)
save_yolo_annot(val_yaml, val_annot_yaml)

print(train_yaml.head(20))
print(train_yaml.loc[train_yaml['name']=='image (10)'])
img=cv2.imread(train_img_yaml+'/image (10).jpg', cv2.IMREAD_COLOR)
img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
cv2.rectangle(img,(217, 101),(453,333),(225,225,255),2)
cv2.rectangle(img,(1,7),(282,333),(0,225,0),2)
plt.imshow(img)
fig=plt.figure()

plt.subplot(1,3,1)
plt.imshow(img[7:333,1:282])
plt.xticks([])
plt.yticks([])
plt.xlabel('Green Line')

plt.subplot(1,3,2)
plt.imshow(img[101:333,217:453])
plt.xticks([])
plt.yticks([])
plt.xlabel('White Line')
plt.show()

def cropping_from_image(path,Data_information):
    cropped_image=[]
    label=[]
    for i in range(0,len(Data_information)):
        img=cv2.imread(path+'/'+Data_information['name'][i]+'.jpg',cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_=img[Data_information['ymin'][i]:Data_information['ymax'][i],Data_information['xmin'][i]:Data_information['xmax'][i]]
        cropped_image.append(img_)
        
        label.append(Data_information['label'][i])
    return cropped_image , label

train_image, train_label = cropping_from_image(train_img_yaml,train_yaml)
test_image, test_label = cropping_from_image(test_img_yaml,test_yaml)
val_image, val_label = cropping_from_image(val_img_yaml,val_yaml)

print(len(train_image) == len(train_label))
print(len(test_image) == len(test_label))
print(len(val_image) == len(val_label))

fig=plt.figure(figsize=(10,10))

for i in range (1,10):
    random=np.random.randint(0,len(train_image))
    plt.subplot(3,3,i)
    plt.imshow(train_image[random])
    plt.xlabel(train_label[random])
    plt.xticks([])
    plt.yticks([])

fig=plt.figure(figsize=(10,10))

for i in range (1,10):
    random=np.random.randint(0,len(test_image))
    plt.subplot(3,3,i)
    plt.imshow(test_image[random])
    plt.xlabel(test_label[random])
    plt.xticks([])
    plt.yticks([])

yaml_content = """
train: C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/train
val: C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/validation
test: C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/test

nc: 2
names: ['person', 'person-like']
"""

yaml_path = 'C:/Users/91776/Desktop/Pedestrian_Detection/pedestrian_detection.yaml'
with open(yaml_path, 'w') as yaml_file:
    yaml_file.write(yaml_content)

model = YOLO("yolo11s.yaml").load("yolo11s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data=yaml_path, epochs=15, imgsz=640, batch=8)

model.save('C:/Users/91776/Desktop/Pedestrian_Detection/transfer_learning.pt')

results = model.val(data=yaml_path)

def perform_inference(model, image):
    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    return results

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): Bounding box coordinates [x1, y1, x2, y2].
        box2 (list): Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        Float: IoU value between the two bounding boxes.
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of each bounding box
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    union_area = area1 + area2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def apply_nms(boxes, scores, iou_threshold=0.5):
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    if len(boxes) == 0:
        return []

    # Sort boxes by their confidence scores
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep_indices = []
    while len(sorted_indices) > 0:
        # Pick the box with the highest confidence score
        best_index = sorted_indices[0]
        keep_indices.append(best_index)

        # Calculate IoU with other boxes
        ious = [calculate_iou(boxes[best_index], boxes[idx]) for idx in sorted_indices[1:]]

        # Remove boxes with IoU higher than the threshold
        sorted_indices = [sorted_indices[i + 1] for i, iou in enumerate(ious) if iou <= iou_threshold]

    return keep_indices

def draw_bounding_boxes(image, results, labels, iou_threshold=0.5):
    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        indices = apply_nms(boxes, scores, iou_threshold)
        
        for idx in indices:
            b = boxes[idx]
            c = class_ids[idx]
            annotator.box_label(b, labels[int(c)])
        
        image_with_boxes = annotator.result()
    return image_with_boxes

def display_predictions():
    # Labels for your classes (example)
    labels = ['person', 'person-like']
    fig = plt.figure(figsize=(15, 15))
    image_dir = test_img_yaml
    
    for index in range(9):
        image = cv2.imread(os.path.join(image_dir, f'image ({np.random.randint(20)}).jpg'))
        results = perform_inference(model, image)
        image_with_boxes = draw_bounding_boxes(image, results, labels)
        
        plt.subplot(3, 3, index+1)
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
    
    plt.show(block=True)


display_predictions()






# train_yaml_data = {}

# for i, d in enumerate(train_xml_data, start =1):
#     data_dict = xmltodict.parse(d)
#     yaml_data = yaml.dump(train_yaml_data, sort_keys=False)
#     train_yaml_data.append(yaml_data)


# if __name__ == "__main__":
    #training code
