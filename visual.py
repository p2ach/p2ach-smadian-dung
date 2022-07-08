import torch
import torchvision.models as models
import matplotlib.pylab as plt

from PIL import Image
from network.deeplabv3.deeplabv3 import *
from build_data import *
from module_list import *
import numpy as np

import ttach as tta


# ++++++++++++++++++++ AI challenge Visualisation +++++++++++++++++++++++++
im_size = [512, 512]
# model1 = DeepLabv3Plus(models.resnet101(), num_classes=5).to('cuda')
# model1.load_state_dict(torch.load('model_weights/dice_loss_epoch100_aichallenge_label5_semi_classmix_reco_0.pth', map_location=torch.device('cuda')))
# model1.eval()
# model2 = DeepLabv3Plus(models.resnet101(), num_classes=5).to('cuda')
# model2.load_state_dict(torch.load('model_weights/best_dice_loss_epoch0_aichallenge_label5_semi_classmix_reco_0_20220618.pth', map_location=torch.device('cuda')))
# model2.eval()
model3 = DeepLabv3Plus(models.resnet101(), num_classes=5).to('cuda')#best
model3.load_state_dict(torch.load('model_weights/v3_dice_loss_epoch80_aichallenge_label5_semi_classmix_reco_191123.pth', map_location=torch.device('cuda')))
model3.eval()
# model4 = DeepLabv3Plus(models.resnet101(), num_classes=5).to('cuda')
# model4.load_state_dict(torch.load('model_weights/no_pretrain_v2_dice_loss_epoch160_aichallenge_label5_semi_classmix_reco_0.pth', map_location=torch.device('cuda')))
# model4.eval()
# model5 = DeepLabv3Plus(models.resnet101(), num_classes=5).to('cuda')
# model5.load_state_dict(torch.load('model_weights/no_pretrain_v2_dice_loss_epoch172_aichallenge_label5_semi_classmix_reco_0.pth', map_location=torch.device('cuda')))
# model5.eval()
# root = 'dataset/aichallenge/test/images/'
# root = 'logging/infer_logging/normal/1/'
root = 'logging/infer_logging/multi_instance/1/'
list_images = os.listdir(root)
print(list_images)

colormap = create_pascal_label_colormap()

transforms1 = tta.Compose(
    [
        tta.HorizontalFlip(),
        # tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
    ]
)
# transform1 = tta.FiveCrops(crop_height=512, crop_width=512)
transforms2 = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
    ]
)
transforms3 = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Multiply(factors=[0.9, 1, 1.1]),
    ]
)

# tta_model = tta.SegmentationTTAWrapper(model1, transforms1, merge_mode='mean')



for img in list_images:
    num_segments = 5
    device = torch.device("cpu")
    # model = DeepLabv3Plus(models.resnet101(), num_classes=5).to(device)
    # visualise image id 961 in validation set

    ori_im = Image.open('dataset/aichallenge/test/images/'+img)
    gt_label = Image.open(root+img)
    # gt_label = Image.open(root+'labeled_images/'+img)
    im_tensor, label_tensor = one_sample_transform(ori_im, gt_label, None,
                                                   scale_size=(1.0, 1.0), augmentation=False)
    im_tensor, label_tensor = im_tensor.to('cuda'), label_tensor.to('cuda')


    im_tensor_512, label_tensor_512 = one_sample_transform(ori_im, gt_label, None,[512,512],
                                                   scale_size=(1.0, 1.0), augmentation=False)
    im_tensor_512, label_tensor_512 = im_tensor_512.to('cuda'), label_tensor_512.to('cuda')



    im_tensor_no_resize, label_tensor_no_resize = no_resize_transform(ori_im, gt_label, None, crop_size=ori_im.size,
                                                   scale_size=(1.0, 1.0), augmentation=False)
    im_tensor_no_resize, label_tenso_no_resize = im_tensor_no_resize.to('cuda'), label_tensor_no_resize.to('cuda')

    logits, _, _ = model3(im_tensor.unsqueeze(0))
    logits1, _, _ = model3(im_tensor.unsqueeze(0))
    logits2, _, _ = model3(im_tensor_512.unsqueeze(0))
    logits3, _, _ = model3(im_tensor.unsqueeze(0))
    logits4, _, _ = model3(im_tensor.unsqueeze(0))


    merger1 = tta.base.Merger(type='mean', n=len(transforms1))
    merger2 = tta.base.Merger(type='mean', n=len(transforms2))
    merger3 = tta.base.Merger(type='mean', n=len(transforms3))

    # for i, p in enumerate(transform1.params):
    #     augmented_image = transform1.apply_aug_image(im_tensor_no_resize.unsqueeze(0), **{transform1.pname: p})
    #     with torch.no_grad():
    #         augmented_output, _, _ = model1(augmented_image)
    #         deaugmented_output = transform1.apply_deaug_mask(augmented_output)
    #         merger1.append(deaugmented_output)

    # for transformer in transforms1:
    #     augmented_image = transformer.augment_image(im_tensor.unsqueeze(0))
    #     # augmented_image = augmented_image.unsqueeze(0)
    #     # augmented_image = torch.squeeze(augmented_image)
    #     with torch.no_grad():
    #         augmented_output, _, _ = model2(augmented_image)
    #         deaugmented_output = transformer.deaugment_mask(augmented_output)
    #         merger1.append(deaugmented_output)
    #
    # logits1 = merger1.result
    #
    # for transformer in transforms1:
    #     augmented_image = transformer.augment_image(im_tensor.unsqueeze(0))
    #     # augmented_image = augmented_image.unsqueeze(0)
    #     # augmented_image = torch.squeeze(augmented_image)
    #     with torch.no_grad():
    #         augmented_output, _, _ = model3(augmented_image)
    #         deaugmented_output = transformer.deaugment_mask(augmented_output)
    #         merger2.append(deaugmented_output)
    #
    # logits2 = merger2.result
    #
    # for transformer in transforms1:
    #     augmented_image = transformer.augment_image(im_tensor.unsqueeze(0))
    #     # augmented_image = augmented_image.unsqueeze(0)
    #     # augmented_image = torch.squeeze(augmented_image)
    #     with torch.no_grad():
    #         augmented_output, _, _ = model4(augmented_image)
    #         deaugmented_output = transformer.deaugment_mask(augmented_output)
    #         merger3.append(deaugmented_output)
    #
    # logits3 = merger3.result

    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)
    bin_count = np.bincount(np_label_reco.reshape(-1))
    bin_count[0] = 0
    found_class_num = bin_count.argmax()
    mask = np.where(np_label_reco == found_class_num, found_class_num, 0)
    mask = cv2.resize(mask, (ori_im.size[0], ori_im.size[1]),
                                                               interpolation=cv2.INTER_NEAREST)
    reco_blend = Image.blend(ori_im, Image.fromarray(color_map(np.array(mask), colormap)), alpha=0.7)

    logits1 = F.interpolate(logits1, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits1, dim=1), dim=1)
    np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)
    bin_count = np.bincount(np_label_reco.reshape(-1))
    bin_count[0] = 0
    found_class_num = bin_count.argmax()
    mask = np.where(np_label_reco == found_class_num, found_class_num, 0)
    mask = cv2.resize(mask, (ori_im.size[0], ori_im.size[1]),
                                                               interpolation=cv2.INTER_NEAREST)
    reco_blend1 = Image.blend(ori_im, Image.fromarray(color_map(np.array(mask), colormap)), alpha=0.7)

    logits2 = F.interpolate(logits2, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits2, dim=1), dim=1)
    np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)
    bin_count = np.bincount(np_label_reco.reshape(-1))
    bin_count[0] = 0
    found_class_num = bin_count.argmax()
    mask = np.where(np_label_reco == found_class_num, found_class_num, 0)
    mask = cv2.resize(mask, (ori_im.size[0], ori_im.size[1]),
                                                               interpolation=cv2.INTER_NEAREST)
    reco_blend2 = Image.blend(ori_im, Image.fromarray(color_map(np.array(mask), colormap)), alpha=0.7)

    logits3 = F.interpolate(logits3, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits3, dim=1), dim=1)
    np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)
    bin_count = np.bincount(np_label_reco.reshape(-1))
    bin_count[0] = 0
    found_class_num = bin_count.argmax()
    mask = np.where(np_label_reco == found_class_num, found_class_num, 0)
    mask = cv2.resize(mask, (ori_im.size[0], ori_im.size[1]),
                                                               interpolation=cv2.INTER_NEAREST)
    reco_blend3 = Image.blend(ori_im, Image.fromarray(color_map(np.array(mask), colormap)), alpha=0.7)

    logits4 = F.interpolate(logits4, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits4, dim=1), dim=1)
    np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)
    bin_count = np.bincount(np_label_reco.reshape(-1))
    bin_count[0] = 0
    found_class_num = bin_count.argmax()
    mask = np.where(np_label_reco == found_class_num, found_class_num, 0)
    mask = cv2.resize(mask, (ori_im.size[0], ori_im.size[1]),
                                                               interpolation=cv2.INTER_NEAREST)
    reco_blend4 = Image.blend(ori_im, Image.fromarray(color_map(np.array(mask), colormap)), alpha=0.7)

    fig, ax = plt.subplots(1, 6, figsize=(10, 6))
    ax[0].imshow(ori_im)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_xlabel('original image')
    ax[1].imshow(reco_blend)
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xlabel('reco_blend')
    ax[2].imshow(reco_blend1)
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    ax[2].set_xlabel('reco_blend2')
    ax[3].imshow(reco_blend2)
    ax[3].set_xticklabels([])
    ax[3].set_yticklabels([])
    ax[3].set_xlabel('reco_blend3')
    ax[4].imshow(reco_blend3)
    ax[4].set_xticklabels([])
    ax[4].set_yticklabels([])
    ax[4].set_xlabel('reco_blend4')
    ax[5].imshow(reco_blend4)
    ax[5].set_xticklabels([])
    ax[5].set_yticklabels([])
    ax[5].set_xlabel('reco_blend5')
    plt.show()
    # plt.savefig('./logging/infer_logging/visual/' + img,
    #             bbox_inches='tight')


    # im_tensor, label_tensor = transform(im, gt_label, None, crop_size=im_size, scale_size=(1.0, 1.0), augmentation=False)
    # im_w, im_h = im.size

# model.load_state_dict(torch.load('model_weights/pascal_label60_sup.pth', map_location=torch.device('cpu')))
# model.eval()
# logits, _ = model(im_tensor.unsqueeze(0))
# logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
# max_logits, label_sup = torch.max(torch.softmax(logits, dim=1), dim=1)
# label_sup[label_tensor == -1] = -1
#
# model.load_state_dict(torch.load('model_weights/pascal_label60_semi_classmix.pth', map_location=torch.device('cpu')))
# model.eval()
# logits, _ = model(im_tensor.unsqueeze(0))
# logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
# max_logits, label_classmix = torch.max(torch.softmax(logits, dim=1), dim=1)
# label_classmix[label_tensor == -1] = -1
#
# model.load_state_dict(torch.load('model_weights/pascal_label60_semi_classmix_reco.pth', map_location=torch.device('cpu')))
# model.eval()
# logits, rep = model(im_tensor.unsqueeze(0))
# logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
# max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
# label_reco[label_tensor == -1] = -1

# fig, ax = plt.subplots(1, 4, figsize=(10, 6))
#
# gt_blend = Image.blend(im, Image.fromarray(color_map(label_tensor[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
# sup_blend = Image.blend(im, Image.fromarray(color_map(label_sup[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
# classmix_blend = Image.blend(im, Image.fromarray(color_map(label_classmix[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
# reco_blend = Image.blend(im, Image.fromarray(color_map(label_reco[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
#
# ax[0].imshow(gt_blend)
# ax[0].set_xticklabels([])
# ax[0].set_yticklabels([])
# ax[0].set_xlabel('Ground Truth')
# ax[1].imshow(sup_blend)
# ax[1].set_xticklabels([])
# ax[1].set_yticklabels([])
# ax[1].set_xlabel('Supervised')
# ax[2].imshow(classmix_blend)
# ax[2].set_xticklabels([])
# ax[2].set_yticklabels([])
# ax[2].set_xlabel('ClassMix')
# ax[3].imshow(reco_blend)
# ax[3].set_xticklabels([])
# ax[3].set_yticklabels([])
# ax[3].set_xlabel('ClassMix + ReCo')


