import os
import json

data_path="dataset/dung/train"
json_path1=os.path.join(data_path,'V-babypoop-9-54.json')
json_path2=os.path.join(data_path,'V-babypoop-10-169.json')
dst_path=os.path.join(data_path,'labels')

list_labeled_images = os.listdir(os.path.join(data_path,'labeled_images'))


# for img in data_path:
    # path2img = os.path.join(data_path,img)

with open(json_path2,'r',encoding='utf-8') as f:
    json_objects=json.load(f)

for obj in json_objects['recog_info']['recog_data']:
    obj

print(json_objects)




