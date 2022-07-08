import torch
import torchvision.models as models
import matplotlib.pylab as plt
from tqdm import tqdm
from PIL import Image, ImageOps
from network.deeplabv3.deeplabv3 import *
from build_data import *
from module_list import *
import os
import pandas as pd
import ttach as tta
import torch
import cv2

import torch.nn.functional as nnf


# temp_test_dataset=['hs28iXxtsb.jpg','0piMLnYCnq.jpg','0upuiXL0IE.jpg','1hxqPDmyLq.jpg','1Jjxy48502.jpg','1kymJY2aDH.jpg','1lOHSLlEKL.jpg','1MZfG352GO.jpg','1WOHtM7KDZ.jpg','2niR9vHK3w.jpg','2qCyoGa8Ss.jpg',
# '2xI54TCpy3.jpg','3BhnkPhTuO.jpg','3MIHNmruvz.jpg','3ZcgUcrFpc.jpg','4bAnMypM6o.jpg','4dMUvNEraj.jpg','4LZruaUqf0.jpg','5cGWCAnPRx.jpg','5XekvWD0Hx.jpg','6aFb5AuxHU.jpg','6TArImG0lw.jpg','6vdwefw70P.jpg',
# '6YSxZbI6xh.jpg','7mmNXHdyHk.jpg','7opJMupzgU.jpg']




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
model2 = DeepLabv3Plus(models.resnet101(), num_classes=num_segments).to('cuda')

test_dataset = "dataset/aichallenge/test/images/"
model.load_state_dict(torch.load('model_weights/dice_loss_epoch100_aichallenge_label5_semi_classmix_reco_0.pth', map_location=torch.device('cuda')))
# model2.load_state_dict(torch.load('model_weights/best_aichallenge_label5_semi_classmix_reco_0.pth', map_location=torch.device('cuda')))
model.eval()

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
        # tta.Scale(scales=[1, 2]),
        # tta.Multiply(factors=[0.9, 1, 1.1]),
        # tta.Multiply(factors=[0.9, 1, 1.1]),
    ]
)

tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')

class_map = {1:'container_truck', 2:'forklift', 3:'reach_stacker', 4:'ship'}
except_list = []
file_names = []
classes=[]
predictions=[]
list_dir = os.listdir(test_dataset)
# list_dir=temp_test_dataset
zero_count=0


multi_class_count=0
multi_instance_count=0
colormap = create_cityscapes_label_colormap()

for i, test_img in tqdm(enumerate(list_dir)):

    RETRY=False

    ori_im = Image.open(test_dataset + test_img)
    im = ori_im.resize((512, 512))
    gt_label = Image.open(test_dataset + test_img)
    im_tensor, label_tensor = one_sample_transform(ori_im, gt_label, None, crop_size=list(im.size).reverse(), scale_size=(1.0, 1.0),augmentation=False)
    im_tensor, label_tensor = im_tensor.to('cuda'), label_tensor.to('cuda')
    # logits, _, _ = model(im_tensor.unsqueeze(0))
    # logits, _ = tta_model(im_tensor.unsqueeze(0))
    # logits, _ = tta_model(im_tensor.permute(1,2,0))
    # batch_im_tensor = torch.unsqueeze(im_tensor,0)

    merger = tta.base.Merger(type='max', n=len(transforms))
    # for transformer in transforms:
    #     augmented_image = transformer.augment_image(im_tensor.unsqueeze(0))
    #     # augmented_image = augmented_image.unsqueeze(0)
    #     # augmented_image = torch.squeeze(augmented_image)
    #     with torch.no_grad():
    #         augmented_output, _,_ = model(augmented_image)
    logits, _,cls_logits = model(im_tensor.unsqueeze(0))
    # #     # if self.output_key is not None:
    # #     #     augmented_output = augmented_output[self.output_key]
    #         deaugmented_output = transformer.deaugment_mask(augmented_output)
    #
    #     try:
    #         merger.append(deaugmented_output)
    #     except:
    #         deaugmented_output=nnf.interpolate(deaugmented_output, size=(merger.output.shape[2], merger.output.shape[3]), mode='bicubic', align_corners=False)
    #         merger.append(deaugmented_output)
    #         pass

    # logits = merger.result

    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    # label_reco[label_tensor == -1] = -1
    np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)

    # np_label_reco = cv2.resize(np_label_reco, (ori_im.size[0], ori_im.size[1]),interpolation=cv2.INTER_NEAREST)
    # np_label_reco=
    # np_mask, uids = color_map_with_id(np_label_reco, colormap)
    # img = Image.fromarray(mat, 'L')




    # preriction img to txt file
    bin_count = np.bincount(np_label_reco.reshape(-1))
    bin_count[0] = 0
    found_class_num = bin_count.argmax()

    found_cls = torch.max(torch.softmax(cls_logits, dim=1), dim=1)
    found_cls = int(found_cls[1].cpu().detach().numpy())

    FLAG_CLASS_WRONG = False

    if found_class_num != int(found_cls)+1:
        print(test_img + " is wrong class, found_cls{} seg found_class_num{}".format(found_cls+1,found_class_num))
        FLAG_CLASS_WRONG=True


    mask = np.where(np_label_reco==found_class_num,255,0)
    mask = np.zeros(mask.shape, np.uint8)

    mask_contours = mask
    count_size=0
    for _s in bin_count:
        if _s>0:
            count_size+=1

    if count_size>1:
        multi_class_count+=1
        # print("{}th {} multi_class_count is {}, bin count".format(i, test_img, multi_class_count), bin_count)
        # fig, ax = plt.subplots(1, 4, figsize=(10, 6))
        # ax[0].imshow(im)
        # ax[1].imshow(np_label_reco, cmap="gray")
        # # ax[2].imshow(mask_contours_img, cmap="gray")
        # ax[3].imshow(gt_label)
        # # plt.show()
        # # ax[0].set_xticklabels([])
        # # ax[0].set_yticklabels([])
        # # ax[0].set_xlabel('{} image result, class {}'.format(test_img, found_class_num))
        # # plt.savefig('./logging/infer_logging/' + test_img[:-4] + 'class_{}.jpg'.format(found_class_num),
        # #             bbox_inches='tight')
        # ax[0].set_xlabel('{} image result, class {}'.format(test_img, found_class_num))
        # plt.savefig(
        #     './logging/infer_logging/multi_output/{}/'.format(found_class_num) + test_img[:-4] + 'class_{}.jpg'.format(
        #         found_class_num),
        #     bbox_inches='tight')


    for _class in range(bin_count.size):
        if bin_count[_class] > 0:



            if _class == found_class_num:
                gray_np_label_reco = np.where(np_label_reco==found_class_num,4,0)
                im2, contour, hierarchy = cv2.findContours(gray_np_label_reco.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if len(contour)> 1:
                    # if count_size > 1:
                    multi_instance_count += 1
                    # print("{}th {} multi_instance_count is {}, number of contours".format(i, test_img, multi_instance_count), len(contour))
                        
                    area_minimum=0
                    area_second_minimum=0
                    prev_mask = np.zeros(gray_np_label_reco.shape, np.uint8)
                    prev_mask_contours = prev_mask


                    for ctr in contour:
                        if ctr.size > 5:
                            _area=cv2.contourArea(ctr)
                            if _area is not None:
                                if _area>area_minimum:
                                    if area_minimum!=0:
                                        prev_mask = mask
                                        prev_mask_contours = mask_contours
                                        area_second_minimum=area_minimum
                                        area_minimum=_area
                                        mask = np.zeros(gray_np_label_reco.shape, np.uint8)
                                        mask_contours = cv2.drawContours(mask, [ctr], 0, (255), -1)


                                    else:
                                        mask = np.zeros(gray_np_label_reco.shape, np.uint8)
                                        mask_contours = cv2.drawContours(mask, [ctr], 0, (255), -1)
                                        prev_mask = np.zeros(gray_np_label_reco.shape, np.uint8)
                                        prev_mask_contours = np.zeros(gray_np_label_reco.shape, np.uint8)
                                        area_minimum=_area
                                elif _area>area_second_minimum:
                                    prev_mask = np.zeros(gray_np_label_reco.shape, np.uint8)
                                    prev_mask_contours = cv2.drawContours(prev_mask, [ctr], 0, (255), -1)
                                    area_second_minimum=_area
                                # else:
                                #     prev_mask = np.zeros(gray_np_label_reco.shape, np.uint8)
                                #     prev_mask_contours = prev_mask
                    if np.max(mask_contours) ==0:
                        found_class_num=0
                    else:
                        # if found_class_num == 2:
                        #     logits_mask = np.where(mask_contours>0,logits.cpu().detach().numpy()[0,found_class_num,...],0)
                        #     logits_prev_mask = np.where(prev_mask_contours>0,logits.cpu().detach().numpy()[0,found_class_num,...],0)
                        #     if np.mean(logits_prev_mask) > np.mean(logits_mask):
                        #         mask_contours=prev_mask_contours
                        #         print("found best contour with confidence")

                        all_prediction_img = Image.fromarray(np_label_reco * 50)

                        # try:
                        if found_class_num !=4:
                            # mask_contours=mask_contours+prev_mask_contours
                            mask_contours_img = Image.fromarray(mask_contours)
                        else:
                            # if np.max(mask_contours)==0:
                                # print("mask_contours",mask_contours)

                            most_big_w=np.max(np.where(mask_contours>0)[1])-np.min(np.where(mask_contours>0)[1])
                            # most_big_h=np.max(np.where(mask_contours>0)[1])-np.min(np.where(mask_contours>0)[1])
                            most_big_bottom_y=np.min(np.where(mask_contours>0)[0])


                            # try:
                            if np.max(prev_mask_contours)>0:
                                second_big_w=np.max(np.where(prev_mask_contours>0)[1])-np.min(np.where(prev_mask_contours>0)[1])
                                second_big_bottom_y = np.min(np.where(prev_mask_contours > 0)[0])
                            else:
                                second_big_w=0
                                second_big_bottom_y=0

                            # second_big_h


                            if most_big_w > 512*0.35 and second_big_w > 512*0.3:
                                if most_big_bottom_y>second_big_bottom_y:
                                    # mask_contours_img = Image.fromarray(mask_contours)
                                    mask_contours=mask_contours
                                    # print("down")
                                else:
                                    # mask_contours_img = Image.fromarray(prev_mask_contours)
                                    mask_contours = prev_mask_contours
                                    # print("up")
                            else:
                                # mask_contours_img = Image.fromarray(mask_contours)
                                # print("most", most_big_w,second_big_w)
                                mask_contours = mask_contours# + prev_mask_contours
                        # except Exception as e:
                        #     print("error occurs", e, test_img)
                        #     pass

                        # mask_contours = cv2.resize(mask_contours, (im.size[0], im.size[1]),
                        #                            interpolation=cv2.INTER_NEAREST)

                        if found_class_num==1:
                            mask_contours = mask_contours+prev_mask_contours

                        mask_contours = cv2.resize(mask_contours, (ori_im.size[0], ori_im.size[1]),
                                                               interpolation=cv2.INTER_NEAREST)
                        # mask_contours_img = Image.fromarray(mask_contours.astype(np.uint8))
                        # mask_contours_img = ImageOps.invert(mask_contours_img)

                        mask_contours = np.where(mask_contours>0,4,0)

                        found_class_np_label_reco = mask_contours

                        mask_contours_img = Image.fromarray(color_map(mask_contours, colormap))
                        reco_blend = Image.blend(ori_im, mask_contours_img, alpha=.3)

                        if FLAG_CLASS_WRONG:
                            fig, ax = plt.subplots(1, 4, figsize=(10, 6))
                            ax[0].imshow(im)
                            ax[1].imshow(all_prediction_img, cmap="gray")
                            ax[2].imshow(mask_contours_img, cmap="gray")
                            ax[3].imshow(reco_blend)
                            # plt.show()
                            # ax[0].set_xticklabels([])
                            # ax[0].set_yticklabels([])
                            # ax[0].set_xlabel('{} image result, class {}'.format(test_img, found_class_num))
                            # plt.savefig('./logging/infer_logging/' + test_img[:-4] + 'class_{}.jpg'.format(found_class_num),
                            #             bbox_inches='tight')
                            ax[0].set_xlabel('{} image result, class {}'.format(test_img, found_class_num))

                            if not os.path.isdir('./logging/infer_logging/multi_instance/{}/'.format(found_class_num)):
                                os.makedirs('./logging/infer_logging/multi_instance/{}/'.format(found_class_num))
                                os.chmod('./logging/infer_logging/multi_instance/{}/'.format(found_class_num),0o777)

                            plt.savefig('./logging/infer_logging/multi_instance/{}/'.format(found_class_num) + test_img[:-4] + '.jpg',
                                        bbox_inches='tight')
                            FLAG_CLASS_WRONG=False
                else:
                    all_prediction_img = Image.fromarray(np_label_reco * 50)
                    found_class_np_label_reco = np.where(np_label_reco == int(found_class_num), 4, 0)
                    # found_class_prediction_img = Image.fromarray(found_class_np_label_reco * 50)
                    found_class_np_label_reco = cv2.resize(found_class_np_label_reco, (ori_im.size[0], ori_im.size[1]),
                                                           interpolation=cv2.INTER_NEAREST)
                    # found_class_np_label_reco_img = Image.fromarray(found_class_np_label_reco)
                    # found_class_np_label_reco_img = ImageOps.invert(found_class_np_label_reco_img)

                    found_class_np_label_reco_img = Image.fromarray(color_map(found_class_np_label_reco, colormap))
                    reco_blend = Image.blend(ori_im, found_class_np_label_reco_img, alpha=.3)

                    if FLAG_CLASS_WRONG:
                        fig, ax = plt.subplots(1, 4, figsize=(10, 6))
                        ax[0].imshow(im)
                        ax[1].imshow(all_prediction_img, cmap="gray")
                        ax[2].imshow(found_class_np_label_reco_img, cmap="gray")
                        ax[3].imshow(reco_blend)
                        # plt.show()
                        ax[0].set_xticklabels([])
                        ax[0].set_yticklabels([])
                        ax[0].set_xlabel('{} image result, class {}'.format(test_img, found_class_num))
                        plt.savefig('./logging/infer_logging/' + test_img[:-4] + 'class_{}.jpg'.format(found_class_num),
                                    bbox_inches='tight')
                        ax[0].set_xlabel('{} image result, class {}'.format(test_img, found_class_num))
                        if not os.path.isdir('./logging/infer_logging/normal/{}/'.format(found_class_num)):
                            os.makedirs('./logging/infer_logging/normal/{}/'.format(found_class_num))
                            os.chmod('./logging/infer_logging/normal/{}/'.format(found_class_num),0o777)

                        plt.savefig('./logging/infer_logging/normal/{}/'.format(found_class_num) + test_img[:-4] + '.jpg',
                                    bbox_inches='tight')
                        FLAG_CLASS_WRONG=False




    # position_mask=np.where(mask_contours>0)
    # 
    # # for pos_y,pos_x in position_mask:
    # try:
    #     if len(position_mask[0])>1:
    #         min_y=np.min(position_mask[0])
    #         max_y=np.max(position_mask[0])
    # 
    #         min_x=np.min(position_mask[1])
    #         max_x=np.max(position_mask[1])
    #         _height = mask_contours.shape[0]
    #         _width = mask_contours.shape[1]
    # 
    #         if (min_x-max_x) < _width*0.1 or  (min_y-max_y) < _height*0.1 or ((min_y-max_y) > _height*0.8 and (min_x-max_x) > _width*0.8):
    #             RETRY=True
    # except:
    #     pass

    # if found_class_num==0 or (RETRY and found_class_num!=4):
    if found_class_num==0:# or (RETRY and found_class_num!=4):
        zero_count+=1
        # print("zero_count is ", zero_count)
        # except_list.append((i,test_img[:-4]+'.jpg'))
        # found_class_num=4
        #
        # all_prediction_img = Image.fromarray(np_label_reco * 50)
        # found_class_np_label_reco = np.where(np_label_reco == int(found_class_num), np_label_reco, 0)
        # found_class_prediction_img = Image.fromarray(found_class_np_label_reco * 50)
        except_list.append((i, test_img[:-4] + '.jpg'))
        # im_mirror = ImageOps.mirror(im)
        im_mirror = im.resize((512, 512))
        im_mirror_tensor, label_tensor = one_sample_transform(im_mirror, gt_label, None,
                                                              crop_size=list(im_mirror.size).reverse(),
                                                              scale_size=(1.0, 1.0), augmentation=False)
        im_mirror_tensor = im_mirror_tensor.to('cuda')
        logits, _, _ = model(im_mirror_tensor.unsqueeze(0))
        logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
        max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
        # label_reco[label_tensor == -1] = -1
        np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)
        np_label_reco = cv2.resize(np_label_reco,(im.size[0],im.size[1]),interpolation=cv2.INTER_NEAREST)


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
        if found_class_num == 0:
            found_class_num=4
            print("zero_count is ", zero_count)
        print("{} found_class_num is ".format(test_img), found_class_num)
        all_prediction_img = Image.fromarray(np_label_reco * 50)
        found_class_np_label_reco = np.where(np_label_reco == int(found_class_num), 3, 0)


        # found_class_prediction_img = Image.fromarray(found_class_np_label_reco * 50)
        mask_cfound_class_np_label_recoontours = cv2.resize(found_class_np_label_reco, (ori_im.size[0], ori_im.size[1]),
                                   interpolation=cv2.INTER_NEAREST)


        mask_contours_img = Image.fromarray(color_map(mask_cfound_class_np_label_recoontours, colormap))
        reco_blend = Image.blend(ori_im, mask_contours_img, alpha=.3)

        if FLAG_CLASS_WRONG:
            fig, ax = plt.subplots(1, 4, figsize=(10, 6))
            ax[0].imshow(im)
            ax[1].imshow(all_prediction_img, cmap="gray")
            ax[2].imshow(mask_contours_img, cmap="gray")
            ax[3].imshow(reco_blend)
            # plt.show()
            # ax[0].set_xticklabels([])
            # ax[0].set_yticklabels([])
            # ax[0].set_xlabel('{} image result, class {}'.format(test_img, found_class_num))
            # plt.savefig('./logging/infer_logging/' + test_img[:-4] + 'class_{}.jpg'.format(found_class_num),
            #             bbox_inches='tight')
            ax[0].set_xlabel('{} image result, class {}'.format(test_img, found_class_num))
            if not os.path.isdir('./logging/infer_logging/low_threshold/{}/'.format(found_class_num)):
                os.makedirs('./logging/infer_logging/low_threshold/{}/'.format(found_class_num))
                os.chmod('./logging/infer_logging/low_threshold/{}/'.format(found_class_num),0o777)
            plt.savefig('./logging/infer_logging/low_threshold/' + test_img[:-4] + '.jpg',
                        bbox_inches='tight')
            FLAG_CLASS_WRONG=False
    # else:
    #     all_prediction_img = Image.fromarray(np_label_reco * 50)
    #     found_class_np_label_reco = np.where(np_label_reco == int(found_class_num), np_label_reco, 0)
    #     found_class_prediction_img = Image.fromarray(found_class_np_label_reco * 50)

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
submission_df.to_csv('./result_output/20220619_ksj_18.csv',index=False, encoding='utf-8')


