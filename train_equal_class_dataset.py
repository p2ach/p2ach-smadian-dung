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
torch.backends.cudnn.enabled = False
transform = T.ToPILImage()

parser = argparse.ArgumentParser(description='Semi-supervised Segmentation with Perfect Labels')
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num_labels', default=14, type=int, help='number of labelled training data, set 0 to use all training data')
# parser.add_argument('--lr', default=2.5e-3, type=float)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--dataset', default='dung', type=str, help='aichallenge, pascal, cityscapes, sun')
parser.add_argument('--apply_aug', default='classmix', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--id', default=1, type=int, help='number of repeated samples')
parser.add_argument('--weak_threshold', default=0.7, type=float)
parser.add_argument('--strong_threshold', default=0.97, type=float)
parser.add_argument('--apply_reco', default=False)
parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--output_dim', default=256, type=int, help='output dimension from representation head')
parser.add_argument('--backbone', default='deeplabv3p', type=str, help='choose backbone: deeplabv3p, deeplabv2')
parser.add_argument('--seed', default=191123, type=int)
parser.add_argument('--batch_size', default=4, type=int)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data_loader = BuildDataLoader(args.dataset, args.num_labels,[512,512],args.batch_size)
train_loader, test_loader = data_loader.build(supervised=False)

# Load Semantic Network
device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
# device = 'cpu'
if args.backbone == 'deeplabv3p':
    model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)
elif args.backbone == 'deeplabv2':
    model = DeepLabv2(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)

# model.load_state_dict(torch.load('model_weights/best_aichallenge_label5_semi_classmix_reco_0.pth', map_location=torch.device('cuda')))
# model.load_state_dict(torch.load('model_weights/v3_dice_loss_epoch7_aichallenge_label5_semi_classmix_reco_191123.pth', map_location=torch.device('cuda')))
# model.eval()
total_epoch = 200
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = PolyLR(optimizer, total_epoch, power=0.9)


train_epoch = len(train_loader)
test_epoch = len(test_loader)

avg_cost = np.zeros((total_epoch, 6))
iteration = 0
for index in range(total_epoch):
    # cost = np.zeros(4)
    train_dataset = iter(train_loader)



    model.train()
    conf_mat = ConfMatrix(data_loader.num_segments)
    l_conf_mat = ConfMatrix(data_loader.num_segments)
    u_conf_mat = ConfMatrix(data_loader.num_segments)
    # for i in range(1):
    for i in tqdm(range(train_epoch)):

        train_data, train_label = train_dataset.next()
        train_data, train_label = train_data.to(device), train_label.to(device)

        optimizer.zero_grad()

        pred, rep = model(train_data)
        pred_large = F.interpolate(pred, size=train_label.shape[1:], mode='bilinear', align_corners=True)
        sup_loss = compute_supervised_loss(pred_large, train_label)

        # regional contrastive loss
        if args.apply_reco:
            with torch.no_grad():
                mask = F.interpolate((train_label.unsqueeze(1) >= 0).float(), size=pred.shape[2:], mode='nearest')
                label = F.interpolate(label_onehot(train_label, data_loader.num_segments), size=pred.shape[2:],mode='nearest')
                prob = torch.softmax(pred, dim=1)

            reco_loss = compute_reco_loss(rep, label, mask, prob, args.strong_threshold, args.temp,
                                          args.num_queries, args.num_negatives)
            loss = sup_loss + reco_loss
        else:
            loss = sup_loss

        loss.backward()
        optimizer.step()

        conf_mat.update(pred_large.argmax(1).flatten(), train_label.flatten())
        avg_cost[index, 0] += loss.item() / train_epoch

        iteration += 1

    avg_cost[index, 1:3] = conf_mat.get_metrics()


    if index ==0 or index % 1 ==0:
        with torch.no_grad():
            model.eval()
            test_dataset = iter(test_loader)
            conf_mat = ConfMatrix(data_loader.num_segments)

            for i in tqdm(range(test_epoch)):

                test_data, test_label = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.to(device)

                pred, _ = model(test_data)
                pred_large = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)

                loss = compute_supervised_loss(pred_large, test_label)

                # compute metrics by confusion matrix
                conf_mat.update(pred_large.argmax(1).flatten(), test_label.flatten())
                avg_cost[index, 3:] += loss.item() / test_epoch
            avg_cost[index, 4:6] = conf_mat.get_metrics()
            # print("conf_mat.get_metrics() : ",conf_mat.get_metrics())


    scheduler.step()
    print(
        'EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f}'
        .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                avg_cost[index][3], avg_cost[index][4], avg_cost[index][5]))
    print('Top: mIoU {:.4f} IoU {:.4f}'.format(avg_cost[:, 4].max(), avg_cost[:, 5].max()))



    # if avg_cost[index][8] >= avg_cost[:, 8].max():
    if args.apply_reco:
        torch.save(model.state_dict(), 'model_weights/v3_dice_loss_epoch{}_{}_label{}_semi_{}_reco_{}.pth'.format(index,args.dataset, args.num_labels, args.apply_aug, args.seed))
    else:
        torch.save(model.state_dict(), 'model_weights/v3_dice_loss_epoch{}{}_label{}_semi_{}_{}.pth'.format(index,args.dataset, args.num_labels, args.apply_aug, args.seed))

    if args.apply_reco:
        np.save('logging/loss_{}_label{}_semi_{}_reco_{}.npy'.format(index,args.dataset, args.num_labels, args.apply_aug, args.seed), avg_cost)

    else:
        np.save('logging/loss_{}_label{}_semi_{}_{}.npy'.format(index,args.dataset, args.num_labels, args.apply_aug, args.seed), avg_cost)


