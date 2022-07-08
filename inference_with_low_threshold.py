import torch
import torchvision.models as models
import matplotlib.pylab as plt
from tqdm import tqdm
from PIL import Image
from network.deeplabv3.deeplabv3 import *
from build_data import *
from module_list import *
import os
import pandas as pd
import ttach as tta
import torch

from PIL import Image, ImageOps
import torch.nn.functional as nnf

def mask_to_coordinates(mask):
    flatten_mask = mask.flatten()
    if flatten_mask.max() == 0:
        return f'0 {len(flatten_mask)}'
    idx = np.where(flatten_mask!=0)[0]
    steps = idx[1:]-idx[:-1]
    new_coord = []
    step_idx = np.where(np.array(steps)!=1)[0]
    start = np.append(idx[0], idx[step_idx+1])
    end = np.append(idx[step_idx], idx[-1])
    length = end - start + 1
    for i in range(len(start)):
        new_coord.append(start[i])
        new_coord.append(length[i])
    new_coord_str = ' '.join(map(str, new_coord))
    return new_coord_str

num_segments = 5
model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments).to('cuda')

dir_empty_objects="logging/infer_logging/"

test_dataset = "dataset/aichallenge/test/images/"
model.load_state_dict(torch.load('model_weights/aichallenge_label5_semi_classmix_reco_0.pth', map_location=torch.device('cuda')))
model.eval()

list_empty_objects = os.listdir(dir_empty_objects)
list_empty_objects = [_file.split('class_')[0]+'.jpg' for _file in list_empty_objects if len(_file.split('class_'))==2]

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.9, 1, 1.1]),
    ]
)

tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')

class_map = {1:'container_truck', 2:'forklift', 3:'reach_stacker', 4:'ship'}
except_list = []
file_names = []
classes=[]
predictions=[]
list_dir = os.listdir(test_dataset)
zero_count=0
for i, test_img in tqdm(enumerate(list_empty_objects)):
    im = Image.open(test_dataset + test_img)
    gt_label = Image.open(test_dataset + test_img)



    im_tensor, label_tensor = one_sample_transform(im, gt_label, None, crop_size=list(im.size).reverse(), scale_size=(1.0, 1.0),augmentation=False)

    im_tensor, label_tensor = im_tensor.to('cuda'), label_tensor.to('cuda')

    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)



    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    # label_reco[label_tensor == -1] = -1
    np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)
    # np_label_reco=
    # np_mask, uids = color_map_with_id(np_label_reco, colormap)
    # img = Image.fromarray(mat, 'L')

    # preriction img to txt file
    bin_count = np.bincount(np_label_reco.reshape(-1))
    bin_count[0] = 0
    found_class_num = bin_count.argmax()

    if found_class_num==0:
        zero_count+=1

        except_list.append((i,test_img[:-4]+'.jpg'))
        # im_mirror = ImageOps.mirror(im)
        im_mirror = im.resize((512,512))
        im_mirror_tensor, label_tensor = one_sample_transform(im_mirror, gt_label, None,
                                                              crop_size=list(im_mirror.size).reverse(),
                                                              scale_size=(1.0, 1.0), augmentation=False)
        im_mirror_tensor = im_mirror_tensor.to('cuda')
        logits, _ = model(im_mirror_tensor.unsqueeze(0))
        logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
        max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
        # label_reco[label_tensor == -1] = -1
        np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)

        # logit_zeros = torch.ones([1, 4, logits.shape[2], logits.shape[3]], dtype=torch.float64,device='cuda')
        # logit_oness = torch.zeros([1, 1, logits.shape[2], logits.shape[3]], dtype=torch.float64,device='cuda')
        # logit_biases = torch.cat([logit_zeros,logit_oness*100],dim=1)
        # logits[0,0,...]=logits[0,0,...]-10
        # max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
        # # label_reco[label_tensor == -1] = -1
        # np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)
        bin_count = np.bincount(np_label_reco.reshape(-1))
        bin_count[0] = 0
        found_class_num = bin_count.argmax()
        print("{} found_class_num is ".format(test_img), found_class_num)

        if found_class_num == 0:
            found_class_num=4

        all_prediction_img = Image.fromarray(np_label_reco * 50)
        found_class_np_label_reco = np.where(np_label_reco == int(found_class_num), np_label_reco, 0)
        found_class_prediction_img = Image.fromarray(found_class_np_label_reco * 50)

        fig, ax = plt.subplots(1, 4, figsize=(10, 6))
        ax[0].imshow(im)
        ax[1].imshow(all_prediction_img, cmap="gray")
        ax[2].imshow(found_class_prediction_img, cmap="gray")
        ax[3].imshow(gt_label)
        # plt.show()
        # ax[0].set_xticklabels([])
        # ax[0].set_yticklabels([])
        ax[0].set_xlabel('{} image result, class {}'.format(test_img, found_class_num))
        plt.savefig('./logging/infer_logging/low_threshold/' + test_img[:-4] + 'class_{}.jpg'.format(found_class_num),
                    bbox_inches='tight')
    else:
        all_prediction_img = Image.fromarray(np_label_reco * 50)
        found_class_np_label_reco = np.where(np_label_reco == int(found_class_num), np_label_reco, 0)
        found_class_prediction_img = Image.fromarray(found_class_np_label_reco * 50)

    class_of_image = class_map[found_class_num]
    converted_coordinate = mask_to_coordinates(found_class_np_label_reco)
    file_names.append(test_img[:-4]+'.jpg')
    classes.append(class_of_image)
    predictions.append(converted_coordinate)

    
    

    


    # prediction = label_reco[0].numpy()*50
    # prediction_img = Image.fromarray(prediction)





# sample_submission = pd.read_csv()
sample_submission = pd.read_csv('sample_submission.csv')
submission_df = pd.DataFrame({'file_name':file_names, 'class':classes, 'prediction':predictions})
submission_df = pd.merge(sample_submission['file_name'],submission_df,left_on='file_name',right_on='file_name',how='left')
submission_df.to_csv('./result_output/20220614_ksj_1206.csv',index=False, encoding='utf-8')


