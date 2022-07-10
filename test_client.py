import requests
import json
from torch import nn
import bentoml
from network.deeplabv3.deeplabv3 import *
from build_data import *
from module_list import *
import torchvision.models as models
from bentoml.io import JSON
from fastapi import FastAPI
import io, base64
from build_data import *
from svc_inference import infer
from app import lambda_handler
import svc_inference


def post_request(encoded_string):
    URL = "http://localhost:3000/predict"
    data = {'img_base64': encoded_string}
    # data = {'img_base64':"encoded_string"}
    r = requests.post(url=URL, data=json.dumps(data))

    pastebin_url = r.text
    print("The pastebin URL is:%s" % pastebin_url)

def run_local(encoded_string):
    # str_encoded_string=encoded_string.decode('ascii')
    pred=lambda_handler({"body":{'img_base64':encoded_string}},context=[])
    # num_segments = 17
    # model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments)
    # model.load_state_dict(torch.load('model_weights/v3_dice_loss_epoch0dung_label17_semi_classmix_191123.pth',
    #                                  map_location=torch.device('cpu')))
    # runner = bentoml.pytorch.get("dungdetection:latest").to_runner()
    # runner.init_local()
    # # dec_encoded_string=str_encoded_string.encode('ascii')
    # # dec_encoded_string=bytes(str_encoded_string,encoding='utf-8')
    # pred = infer(encoded_string,runner)
    return pred

if __name__=="__main__":
    list_test_imgs = os.listdir("dataset/dung/test/labeled_images")
    imgs_list=[]
    for img in list_test_imgs:
        try:
            with open("dataset/dung/test/labeled_images/"+img, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('ascii')
            # print(len(encoded_string))
            pred=run_local(encoded_string)
            np_mask=np.array(Image.open("dataset/dung/test/labels/"+img).convert('RGB'))
            for uni_id in np.unique(np_mask):
                if uni_id !=0:
                    if uni_id in svc_inference.class_map:
                        ans='OK'
                    else:
                        ans = 'FAIL'

            print("result, labels : ", ans, np.unique(np_mask),json.loads(pred['body'])['code'])
            imgs_list.append(img.split('.')[0])

        except Exception as e:
            print("pass to this image")
    print("imgs_list : ", imgs_list)


        # post_request(encoded_string)