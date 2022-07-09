import json
import bentoml
from network.deeplabv3.deeplabv3 import *
import torchvision.models as models
from build_data import *
from svc_inference import infer

def lambda_handler(event, context):
    # 1. s3 put event 발생시
    img_base64 = event["img_base64"]
    num_segments = 17
    model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments)
    model.load_state_dict(torch.load('model_weights/v3_dice_loss_epoch0dung_label17_semi_classmix_191123.pth',
                                     map_location=torch.device('cpu')))
    runner = bentoml.pytorch.get("dungdetection:latest").to_runner()
    runner.init_local()
    code=infer(img_base64,runner)
    pred={'code':code}
    return {
        'statusCode': 200,
        'body': json.dumps(pred)
    }