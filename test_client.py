import requests
import json
from io import BytesIO
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
    URL = "https://gxciuxktg6.execute-api.ap-northeast-2.amazonaws.com/prod/dung"
    data = {'img_base64': encoded_string}
    # data = {'img_base64':"encoded_string"}
    r = requests.post(url=URL, data=json.dumps(data))
    return r.json().get('body')


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

            img_rgb = Image.open("dataset/dung/test/labeled_images/"+img).convert('RGB')
            img_rgb=img_rgb.resize((448,448))
            buffered = BytesIO()
            img_rgb.save(buffered, format="PNG")
            encoded_string = base64.b64encode(buffered.getvalue()).decode('ascii')


            # with open("dataset/dung/test/labeled_images/"+img, "rb") as image_file:
            #     encoded_string = base64.b64encode(image_file.read()).decode('ascii')
            # print(len(encoded_string))

            if False:
                pred=run_local(encoded_string)
                code_result=json.loads(pred['body'])['code']
            else:
                pred = post_request(encoded_string)
                code_result = json.loads(pred)['code']

            np_mask=np.array(Image.open("dataset/dung/test/labels/"+img).convert('RGB'))
            for uni_id in np.unique(np_mask):
                if uni_id !=0:
                    label_code=svc_inference.class_map[uni_id]
                    if svc_inference.class_map[uni_id] in code_result:
                        ans = 'OK'
                    else:
                        ans = 'FAIL'
                        break

            if ans == 'OK' and len(np.unique(np_mask)[1:])==len(code_result):
                imgs_list.append(img.split('.')[0])

            print("result, labels : ", ans, np.unique(np_mask),code_result)


        except Exception as e:
            print("pass to this image",e)
    print("imgs_list : ", imgs_list)


        # post_request(encoded_string)