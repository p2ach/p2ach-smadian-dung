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

test_dataset = "dataset/aichallenge/test/images/"
model.load_state_dict(torch.load('model_weights/aichallenge_label5_semi_classmix_reco_0.pth', map_location=torch.device('cuda')))
model.eval()




transforms = tta.Compose(
    [
        # tta.HorizontalFlip(),
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
for i, test_img in tqdm(enumerate(list_dir)):
    im = Image.open(test_dataset + test_img)
    gt_label = Image.open(test_dataset + test_img)
    im_tensor, label_tensor = one_sample_transform(im, gt_label, None, crop_size=list(im.size).reverse(), scale_size=(1.0, 1.0),augmentation=False)
    im_tensor, label_tensor = im_tensor.to('cuda'), label_tensor.to('cuda')

    # logits, _ = tta_model(im_tensor.unsqueeze(0))
    # logits, _ = tta_model(im_tensor.permute(1,2,0))
    # batch_im_tensor = torch.unsqueeze(im_tensor,0)

    merger = tta.base.Merger(type='mean', n=len(transforms))
    for transformer in transforms:
        augmented_image = transformer.augment_image(im_tensor.unsqueeze(0))
        # augmented_image = torch.squeeze(augmented_image)
        with torch.no_grad():
            augmented_output, _ = model(augmented_image)
        # if self.output_key is not None:
        #     augmented_output = augmented_output[self.output_key]
        deaugmented_output = transformer.deaugment_mask(augmented_output)
        try:
            merger.append(deaugmented_output)
        except:

            deaugmented_output=nnf.interpolate(deaugmented_output, size=(merger.output.shape[2], merger.output.shape[3]), mode='bicubic', align_corners=False)
            merger.append(deaugmented_output)
            pass

    logits = merger.result

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
        except_list.append((i,test_img[:-4]+'.jpg'))
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
        plt.savefig('./logging/infer_logging/' + test_img[:-4] + 'class_{}.jpg'.format(found_class_num),
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


