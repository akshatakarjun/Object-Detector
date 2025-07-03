
# from ultralytics import YOLO
# import cv2
# import os
# import glob
# import pandas as pd

# model = YOLO('yolo11s.pt')
# img_source = r"C:\Users\91776\Desktop\Pedestrian_Detection\Test\Test\JPEGImages"
# img_destination = r"C:\Users\91776\Desktop\Pedestrian_Detection\Test\Test\results_existing_weight"

# i_s = r'C:\Users\91776\Desktop\Pedestrian_Detection\trial'
# i_d = r'C:\Users\91776\Desktop\Pedestrian_Detection'

# images = glob.glob(i_s+'/*.*')
# all_dfs = []
# for i, img in enumerate(images, start =1):
#     try:
#         results = model(img)
#         print(f"Type of results: {type(results)}")
        
#         for result in results:
#             output_filename = os.path.join(i_d, f'result_{i}.jpg')
#             result.save(filename=output_filename)
#             print(f'Saved: {output_filename}')
#             df = result.to_df()
#             df['image'] = result.path
#             all_dfs.append(df)
            
        
        
    


            # boxes = result.boxes
            # masks = result.masks
            # keypoints = result.keypoints
            # probs = result.probs
            # obb = result.obb
            # result.show()
            
    # except:
    #     print(f'{img} is not saved')

# final_df = pd.concat(all_dfs, ignore_index=True)
# print(all_dfs)



# model.info()
# print(model.model)


# from IPython.display import display, Image
# from IPython import display
# display.clear_output()

# from PIL import Image

# Load an image
# image = Image.open("C:/Users/91776/Desktop/Pedestrian_Detection/trial/image_(1).jpg")

# Get the size of the image
# width, height = image.size

# print(f"The image resolution is: {width}x{height}")

# image.close()




import os
import os

# folder_path = 'C:/Users/91776/Desktop/Pedestrian_Detection/xml/Test/Annotations'
folder_path = 'C:/Users/91776/Desktop/Pedestrian_Detection/yolo_dataset/train/labels'

# Get all .jpg files
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

# Rename images to 'image (1).jpg', 'image (2).jpg', ...
for idx, filename in enumerate(image_files, start=1):
    new_name = f"image ({idx}).txt"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)

print("Images renamed successfully.")

