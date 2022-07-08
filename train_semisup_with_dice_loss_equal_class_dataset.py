import os

import torch
import torchvision.models as models
import torch.optim as optim
import argparse
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from network.deeplabv3.deeplabv3 import *
from network.deeplabv2 import *
from build_data import *
from module_list import *
import cv2

transform = T.ToPILImage()

parser = argparse.ArgumentParser(description='Semi-supervised Segmentation with Perfect Labels')
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num_labels', default=5, type=int, help='number of labelled training data, set 0 to use all training data')
# parser.add_argument('--lr', default=2.5e-3, type=float)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--dataset', default='aichallenge', type=str, help='aichallenge, pascal, cityscapes, sun')
parser.add_argument('--apply_aug', default='classmix', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--id', default=1, type=int, help='number of repeated samples')
parser.add_argument('--weak_threshold', default=0.7, type=float)
parser.add_argument('--strong_threshold', default=0.97, type=float)
parser.add_argument('--apply_reco', default=True)
parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--output_dim', default=256, type=int, help='output dimension from representation head')
parser.add_argument('--backbone', default='deeplabv3p', type=str, help='choose backbone: deeplabv3p, deeplabv2')
parser.add_argument('--seed', default=191123, type=int)
parser.add_argument('--batch_size', default=16, type=int)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data_loader = BuildDataLoader(args.dataset, args.num_labels,[512,512],args.batch_size)
train_l_loader, train_u_loader, test_sample_loader, test_loader, container_truck_train_l_loader, container_truck_train_u_loader, forklift_train_l_loader, forklift_train_u_loader, reach_stacker_train_l_loader, reach_stacker_train_u_loader, ship_train_l_loader, ship_train_u_loader = data_loader.build(supervised=False)

# Load Semantic Network
device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
if args.backbone == 'deeplabv3p':
    model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)
elif args.backbone == 'deeplabv2':
    model = DeepLabv2(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)

# model.load_state_dict(torch.load('model_weights/best_aichallenge_label5_semi_classmix_reco_0.pth', map_location=torch.device('cuda')))
model.load_state_dict(torch.load('model_weights/v3_dice_loss_epoch7_aichallenge_label5_semi_classmix_reco_191123.pth', map_location=torch.device('cuda')))
model.eval()
total_epoch = 200
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = PolyLR(optimizer, total_epoch, power=0.9)
ema = EMA(model, 0.99)  # Mean teacher model

train_epoch = len(train_l_loader)
test_epoch = len(test_loader)
test_sample_epoch = len(test_sample_loader)
avg_cost = np.zeros((total_epoch, 11))
iteration = 0
for index in range(total_epoch):
    cost = np.zeros(4)
    train_l_dataset = iter(train_l_loader)
    train_u_dataset = iter(train_u_loader)

    container_truck_train_l_dataset = iter(container_truck_train_l_loader)
    container_truck_train_u_dataset = iter(container_truck_train_u_loader)

    forklift_train_l_dataset = iter(forklift_train_l_loader)
    forklift_train_u_dataset = iter(forklift_train_u_loader)

    reach_stacker_train_l_dataset = iter(reach_stacker_train_l_loader)
    reach_stacker_train_u_dataset = iter(reach_stacker_train_u_loader)

    ship_train_l_dataset = iter(ship_train_l_loader)
    ship_train_u_dataset = iter(ship_train_u_loader)

    model.train()
    ema.model.train()
    l_conf_mat = ConfMatrix(data_loader.num_segments)
    u_conf_mat = ConfMatrix(data_loader.num_segments)
    # for i in range(1):
    for i in tqdm(range(train_epoch)):
        train_l_data, train_l_label, train_l_cls_label = train_l_dataset.next()
        train_l_data, train_l_label, train_l_cls_label = train_l_data.to(device), train_l_label.to(device), train_l_cls_label.to(device)

        train_u_data, train_u_label, train_u_cls_label = train_u_dataset.next()
        train_u_data, train_u_label, train_u_cls_label = train_u_data.to(device), train_u_label.to(device), train_u_cls_label.to(device)
        # container_truck_train_l_data, container_truck_train_l_label, container_truck_train_l_cls_label = container_truck_train_l_dataset.next()
        # container_truck_train_l_data, container_truck_train_l_label, container_truck_train_l_cls_label = container_truck_train_l_data.to(device), container_truck_train_l_label.to(device), container_truck_train_l_cls_label.to(device)
        #
        # container_truck_train_u_data, container_truck_train_u_label, container_truck_train_u_cls_label = container_truck_train_u_dataset.next()
        # container_truck_train_u_data, container_truck_train_u_label, container_truck_train_u_cls_label = container_truck_train_u_data.to(device), container_truck_train_u_label.to(device), container_truck_train_u_cls_label.to(device)
        #
        #
        # forklift_train_l_data, forklift_train_l_label, forklift_train_l_cls_label = forklift_train_l_dataset.next()
        # forklift_train_l_data, forklift_train_l_label, forklift_train_l_cls_label = forklift_train_l_data.to(device), forklift_train_l_label.to(device), forklift_train_l_cls_label.to(device)
        #
        # forklift_train_u_data, forklift_train_u_label, forklift_train_u_cls_label = forklift_train_u_dataset.next()
        # forklift_train_u_data, forklift_train_u_label, forklift_train_u_cls_label = forklift_train_u_data.to(device), forklift_train_u_label.to(device), forklift_train_u_cls_label.to(device)
        #
        #
        # reach_stacker_train_l_data, reach_stacker_train_l_label, reach_stacker_train_l_cls_label = reach_stacker_train_l_dataset.next()
        # reach_stacker_train_l_data, reach_stacker_train_l_label, reach_stacker_train_l_cls_label = reach_stacker_train_l_data.to(device), reach_stacker_train_l_label.to(device), reach_stacker_train_l_cls_label.to(device)
        #
        # reach_stacker_train_u_data, reach_stacker_train_u_label, reach_stacker_train_u_cls_label = reach_stacker_train_u_dataset.next()
        # reach_stacker_train_u_data, reach_stacker_train_u_label, reach_stacker_train_u_cls_label = reach_stacker_train_u_data.to(device), reach_stacker_train_u_label.to(device), reach_stacker_train_u_cls_label.to(device)
        #
        #
        # ship_train_l_data, ship_train_l_label, ship_train_l_cls_label = ship_train_l_dataset.next()
        # ship_train_l_data, ship_train_l_label, ship_train_l_cls_label = ship_train_l_data.to(device), ship_train_l_label.to(device), ship_train_l_cls_label.to(device)
        #
        # ship_train_u_data, ship_train_u_label, ship_train_u_cls_label = ship_train_u_dataset.next()
        # ship_train_u_data, ship_train_u_label, ship_train_u_cls_label = ship_train_u_data.to(device), ship_train_u_label.to(device), ship_train_u_cls_label.to(device)
        #
        #
        #
        # train_l_data=torch.cat((container_truck_train_l_data,forklift_train_l_data,reach_stacker_train_l_data,ship_train_l_data),0)
        # train_l_label=torch.cat((container_truck_train_l_label,forklift_train_l_label,reach_stacker_train_l_label,ship_train_l_label),0)
        # train_l_cls_label=torch.cat((container_truck_train_l_cls_label,forklift_train_l_cls_label,reach_stacker_train_l_cls_label,ship_train_l_cls_label),0)
        #
        # train_u_data=torch.cat((container_truck_train_u_data,forklift_train_u_data,reach_stacker_train_u_data,ship_train_u_data),0)
        # train_u_label=torch.cat((container_truck_train_u_label,forklift_train_u_label,reach_stacker_train_u_label,ship_train_u_label),0)
        # train_u_cls_label=torch.cat((container_truck_train_u_cls_label,forklift_train_u_cls_label,reach_stacker_train_u_cls_label,ship_train_u_cls_label),0)
        #
        # dim = 0
        # idx = torch.randperm(train_l_data.shape[dim])
        # train_l_data = train_l_data[idx]
        # train_l_label = train_l_label[idx]
        # train_l_cls_label = train_l_cls_label[idx]
        #
        #
        # dim = 0
        # idx = torch.randperm(train_u_data.shape[dim])
        # train_u_data = train_u_data[idx]
        # train_u_label = train_u_label[idx]
        # train_u_cls_label = train_u_cls_label[idx]





        optimizer.zero_grad()

        # generate pseudo-labels
        with torch.no_grad():
            pred_u, _, cls_fier_u = ema.model(train_u_data)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_label.shape[1:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # random scale images first
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_data, pseudo_labels, pseudo_logits,
                                data_loader.crop_size, data_loader.scale_size, apply_augmentation=False)

            # apply mixing strategy: cutout, cutmix or classmix
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=args.apply_aug)

            # apply augmentation: color jitter + flip + gaussian blur
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                data_loader.crop_size, (1.0, 1.0), apply_augmentation=True)

        # generate labelled and unlabelled data loss
        pred_l, rep_l, cls_fier_l = model(train_l_data)
        pred_l_large = F.interpolate(pred_l, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

        pred_u, rep_u, cls_fier_a = model(train_u_aug_data)
        pred_u_large = F.interpolate(pred_u, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        # supervised-learning loss (dice loss)
        sup_loss = compute_supervised_loss(pred_l_large, train_l_label)
        # sup_loss = dice_loss(pred_l_large, train_l_label)
        # sup_loss = giouloss(pred_l_large, train_l_label)

        
        # unsupervised-learning loss
        unsup_loss = compute_unsupervised_loss(pred_u_large, train_u_aug_label, train_u_aug_logits, args.strong_threshold)

        # apply regional contrastive loss
        if args.apply_reco:
            with torch.no_grad():
                train_u_aug_mask = train_u_aug_logits.ge(args.weak_threshold).float()
                mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                label_l = F.interpolate(label_onehot(train_l_label, data_loader.num_segments), size=pred_all.shape[2:], mode='nearest')
                label_u = F.interpolate(label_onehot(train_u_aug_label, data_loader.num_segments), size=pred_all.shape[2:], mode='nearest')
                label_all = torch.cat((label_l, label_u))

                prob_l = torch.softmax(pred_l, dim=1)
                prob_u = torch.softmax(pred_u, dim=1)
                prob_all = torch.cat((prob_l, prob_u))

            reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, args.strong_threshold,args.temp, args.num_queries, args.num_negatives)
        else:
            reco_loss = torch.tensor(0.0)

        cross_entropy_loss = nn.CrossEntropyLoss()
        # cls_loss_u = cross_entropy_loss(cls_fier_u, train_u_cls_label)
        # cls_loss_l = cross_entropy_loss(cls_fier_l, train_l_cls_label)
        # cls_loss_a = cross_entropy_loss(cls_fier_a, train_u_cls_label)
        # cls_loss = cls_loss_l + cls_loss_a


        loss = sup_loss + unsup_loss + reco_loss#+cls_loss
        loss.backward()
        optimizer.step()
        ema.update(model)

        l_conf_mat.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())
        u_conf_mat.update(pred_u_large_raw.argmax(1).flatten(), train_u_label.flatten())

        cost[0] = sup_loss.item()
        cost[1] = unsup_loss.item()
        cost[2] = reco_loss.item()
        cost[3] = reco_loss.item()
        # cost[3] = cls_loss.item()
        avg_cost[index, :4] += cost / train_epoch
        iteration += 1

    avg_cost[index, 4:6] = l_conf_mat.get_metrics()
    avg_cost[index, 6:8] = u_conf_mat.get_metrics()


    if index ==0 or index % 1 ==0:
        with torch.no_grad():
            ema.model.eval()
            test_dataset = iter(test_loader)
            conf_mat = ConfMatrix(data_loader.num_segments)
            aml_count_multi_classes=0
            aml_count_of_best_class=0
            aml_count_of_empty_class=0
            list_difficult_imgs = []

            for i in tqdm(range(test_epoch)):
                if i>500:
                    break
                test_image, test_data, test_label, cls_label = test_dataset.next()
                test_image = test_image[0]
                test_image = Image.fromarray(test_image.cpu().detach().numpy())
                test_data, test_label, cls_label= test_data.to(device), test_label.to(device), cls_label.to(device)

                pred, rep, clsfier = ema.model(test_data)
                pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
                # loss = compute_supervised_loss(pred, test_label)

                # clsfier

                max_logits, label_reco = torch.max(torch.softmax(pred, dim=1), dim=1)
                colormap = create_cityscapes_label_colormap()
                np_label_reco = label_reco[0].cpu().detach().numpy()
                im = transform(test_data[0])
                # im_label_reco = Image.fromarray(color_map(np_label_reco,colormap))
                # np_label_reco=np.transpose(np_label_reco,(1,0))
                np_mask, uids = color_map_with_id(np_label_reco, colormap)
                im_mask = Image.fromarray(np_mask)
                reco_blend = Image.blend(test_image, im_mask, alpha=.3)

                bin_count = np.bincount(np_label_reco.reshape(-1))
                bin_count[0] = 0
                found_class_num = bin_count.argmax()
                count_multi_classes=sum(_ele > 0 for _ele in bin_count)
                if count_multi_classes>1:
                    aml_count_multi_classes += 1

                count_of_best_class = 0


                if count_multi_classes ==0:
                    aml_count_of_empty_class+=1

                for _class in range(bin_count.size):
                    if bin_count[_class] > 0:
                        if _class == found_class_num:
                            gray_np_label_reco = np.where(np_label_reco == found_class_num, 255, 0)
                            im2, contour, hierarchy = cv2.findContours(gray_np_label_reco.astype(np.uint8),
                                                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            count_of_best_class = len(contour)

                if count_of_best_class> 1:
                    aml_count_of_best_class += 1


                if count_multi_classes==0 or count_multi_classes>1 or count_of_best_class>1:
                    list_difficult_imgs.append(i)


                if len(uids)>1 or len(uids)==0:
                    # print("epoch {}, {}th image uids :".format(index, i), uids)
                    if not os.path.isdir("logging/train_logging/curated_check/"):
                        os.makedirs("logging/train_logging/curated_check/")
                        os.chmod("logging/train_logging/curated_check/",0o777)

                    test_image.save("logging/train_logging/curated_check/curated_{}_{}_im.png".format(index, i))
                    im_mask.save("logging/train_logging/curated_check/curated_{}_{}_mask.png".format(index, i))
                    reco_blend.save("logging/train_logging/curated_check/curated_{}_{}_model_reco.png".format(index, i))

                    np_pred = pred.cpu().detach().numpy()
                    np_test_label = test_label.cpu().detach().numpy()
                    pred_argmax = pred.argmax(1).flatten().cpu().detach().numpy()
                    test_label_argmax = test_label.flatten().cpu().detach().numpy()

                # conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())
                # avg_cost[index, 7] += loss.item() / test_epoch

            # avg_cost[index, 8:] = conf_mat.get_metrics()
            print("list_difficult_imgs :", list_difficult_imgs)

    with torch.no_grad():
        ema.model.eval()
        test_sample_dataset = iter(test_sample_loader)
        conf_mat = ConfMatrix(data_loader.num_segments)





        for i in range(test_sample_epoch):
            test_image, test_data, test_label, cls_label = test_sample_dataset.next()
            test_image=test_image[0]
            test_image=Image.fromarray(test_image.cpu().detach().numpy())
            test_data, test_label, cls_label = test_data.to(device), test_label.to(device), cls_label.to(device)

            pred, rep, cls_prediction = ema.model(test_data)
            pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
            loss = compute_supervised_loss(pred, test_label)

            max_logits, label_reco = torch.max(torch.softmax(pred, dim=1), dim=1)
            colormap = create_cityscapes_label_colormap()
            np_label_reco = label_reco[0].cpu().detach().numpy()
            np_cls_label = cls_label[0].cpu().detach().numpy()
            im = transform(test_data[0])
            # im_label_reco = Image.fromarray(color_map(np_label_reco,colormap))
            # np_label_reco=np.transpose(np_label_reco,(1,0))
            np_mask, uids = color_map_with_id(np_label_reco, colormap)
            im_mask = Image.fromarray(np_mask)
            reco_blend = Image.blend(test_image, im_mask, alpha=.3)
            print("sample_epoch {}, label:{} sample image uids :".format(index,int(np_cls_label)+1),uids)
            
            # test_image.save("logging/sample_epoch:{}_label:{}_im.png".format(index,int(np_cls_label)+1))
            # im_mask.save("logging/sample_epoch:{}_label:{}_mask.png".format(index,int(np_cls_label)+1))
            # reco_blend.save("logging/sample_epoch:{}_label:{}_reco.png".format(index,int(np_cls_label)+1))

            np_pred=pred.cpu().detach().numpy()
            np_test_label=test_label.cpu().detach().numpy()
            pred_argmax=pred.argmax(1).flatten().cpu().detach().numpy()
            test_label_argmax=test_label.flatten().cpu().detach().numpy()

            conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())
            avg_cost[index, 8] += loss.item() / test_epoch

        avg_cost[index, 9:] = conf_mat.get_metrics()

    scheduler.step()
    if index==0 or index % 1 == 0:
        print(
            'EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} {:.4f}, {:.4f} {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} || count of detected multi_classes: {}, contours : {}, empty_class: {}'
            .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                    avg_cost[index][3], avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7],
                    avg_cost[index][8],  # avg_cost[index][8],
                    avg_cost[index][9], avg_cost[index][10], aml_count_multi_classes, aml_count_of_best_class, aml_count_of_empty_class))
        # print('Top: mIoU {:.4f} Acc {:.4f}'.format(avg_cost[:, 8].max(), avg_cost[:, 9].max()))
        print('Top: mIoU {:.4f} Acc {:.4f}'.format(avg_cost[:, 9].max(), avg_cost[:, 10].max()))
    elif index<10:
        print(
            'EPOCH: {:04d} ITER: '
            ''
            ''
            '{:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} {:.4f}, {:.4f} {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f}'
                .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                        avg_cost[index][3], avg_cost[index][4], avg_cost[index][5], avg_cost[index][6],
                        avg_cost[index][7],
                        avg_cost[index][8],  # avg_cost[index][8],
                        avg_cost[index][9], avg_cost[index][10]))
    else:
        print(
            'EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} {:.4f}, {:.4f} {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} || count of detected multi_classes: {}, contours : {}, empty_class: {}'
                .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                        avg_cost[index][3], avg_cost[index][4], avg_cost[index][5], avg_cost[index][6],
                        avg_cost[index][7],
                        avg_cost[index][8],  # avg_cost[index][8],
                        avg_cost[index][9], avg_cost[index][10], aml_count_multi_classes, aml_count_of_best_class, aml_count_of_empty_class))
        # print('Top: mIoU {:.4f} Acc {:.4f}'.format(avg_cost[:, 8].max(), avg_cost[:, 9].max()))
        print('Top: mIoU {:.4f} Acc {:.4f}'.format(avg_cost[:, 9].max(), avg_cost[:, 10].max()))

    # if avg_cost[index][8] >= avg_cost[:, 8].max():
    if args.apply_reco:
        torch.save(ema.model.state_dict(), 'model_weights/v3_dice_loss_epoch{}_{}_label{}_semi_{}_reco_{}.pth'.format(index,args.dataset, args.num_labels, args.apply_aug, args.seed))
    else:
        torch.save(ema.model.state_dict(), 'model_weights/v3_dice_loss_epoch{}{}_label{}_semi_{}_{}.pth'.format(index,args.dataset, args.num_labels, args.apply_aug, args.seed))

    if args.apply_reco:
        np.save('logging/dice_loss_{}_label{}_semi_{}_reco_{}.npy'.format(index,args.dataset, args.num_labels, args.apply_aug, args.seed), avg_cost)

    else:
        np.save('logging/dice_loss_{}_label{}_semi_{}_{}.npy'.format(index,args.dataset, args.num_labels, args.apply_aug, args.seed), avg_cost)


