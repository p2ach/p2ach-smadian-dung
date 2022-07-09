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
num_segments = 17
model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments)
model.load_state_dict(torch.load('model_weights/v3_dice_loss_epoch0dung_label17_semi_classmix_191123.pth', map_location=torch.device('cpu')))

runner = bentoml.pytorch.get("dungdetection:latest").to_runner()
dung_svc = bentoml.Service("dungdetecter", runners=[runner])
# runner.init_local()
# runner.run(pd.DataFrame("/path/to/csv"))

fastapi_app = FastAPI()
@fastapi_app.get("/")
def health_check():
    return {"status": "OK"}

dung_svc.mount_asgi_app(fastapi_app)
@dung_svc.api(input=JSON(), output=JSON())
def predict(input_json):
    list_result_outputs=infer(input_json['img_base64'],runner)
    #
    # img=Image.open(io.BytesIO(base64.decodebytes(bytes(input_json['img_base64']))))
    # im_tensor = one_sample_transform_nolabel(img, None, crop_size=list(img.size).reverse(),
    #                                                scale_size=(1.0, 1.0), augmentation=False)
    # print("im_tensor : ", im_tensor)
    # im_tensor = im_tensor.to('cpu')
    # result = runner.run(im_tensor.unsqueeze(0))

    print(list_result_outputs)
    return {'status': 200, 'output': list_result_outputs}
