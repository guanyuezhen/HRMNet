import sys

sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os, time
import numpy as np
import torch.optim.lr_scheduler
from torch.nn.parallel import gather
from argparse import ArgumentParser
import datasets.dataset as myDataLoader
import datasets.Transforms as myTransforms
from models.model import get_model
from utils.metric_tool import ConfuseMatrixMeter



def label_edge_prediction(label):
    ero = 1 - F.max_pool2d(1 - label, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(label, kernel_size=5, stride=1, padding=2)  # dilation

    edge = dil - ero
    return edge


def BCEDiceLoss(pres, gts):
    bce = F.binary_cross_entropy(pres, gts)
    inter = (pres * gts).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (pres.sum() + gts.sum() + eps)

    return bce + 1 - dice


class DeepSupervisionLoss(nn.Module):
    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()

    def forward(self, pres, gts):
        if isinstance(gts, tuple):
            gts = gts[0]
        b, n, h, w = pres.size()
        loss = 0
        for i in range(n):
            loss += BCEDiceLoss(pres[:, i:i + 1, :, :], gts)
        return loss


@torch.no_grad()
def val(args, val_loader, model, criterion):
    model.eval()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        start_time = time.time()
        if args.onGPU == True:
            img = img.cuda()
            target = target.cuda()
        img_var = torch.autograd.Variable(img).float()
        target_var = torch.autograd.Variable(target).float()
        # run the model
        change_mask, uncertainty_mask = model(img_var)

        pred = torch.round(change_mask[:, 0:1]).long()

        time_taken = time.time() - start_time
        loss = 0

        # compute the confusion matrix
        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())
        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f loss: %.3f time: %.3f' % (iter, total_batches, f1, loss, time_taken),
                  end='')

    average_epoch_loss_val = 0
    scores = salEvalVal.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.):
    # switch to train mode
    model.train()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    for iter, batched_inputs in enumerate(train_loader):

        img, target = batched_inputs
        #
        start_time = time.time()
        if args.onGPU == True:
            img = img.cuda()
            target = target.cuda()
        img_var = torch.autograd.Variable(img).float()
        target_var = torch.autograd.Variable(target).float()
        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)
        # run the mdoel
        change_mask, uncertainty_mask = model(img_var)
        change_pre = change_mask[:, 0:1].detach()
        uncertainty_gt = torch.mul(target_var, (1.0 - change_pre)) + torch.mul(change_pre, (1.0 - target_var))
        loss = criterion(change_mask, target_var) + F.binary_cross_entropy(uncertainty_mask, uncertainty_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.round(change_mask[:, 0:1]).long()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter - cur_iter) * time_taken / 3600
        # Computing F-measure and IoU on GPU
        with torch.no_grad():
            f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

        if iter % 5 == 0:
            print('\riteration: [%d/%d] f1: %.3f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * args.max_epochs, f1, lr, loss.data.item(),
                res_time), end='')

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    return average_epoch_loss_train, scores, lr


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_val_change_detection(args):
    global get_model
    torch.backends.cudnn.benchmark = True
    SEED = args.seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # formulate models
    model = get_model()

    args.save_dir = args.save_dir + args.model_name + '/' + args.file_root + '_iter_' + str(
        args.max_steps) + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + '/'

    if args.file_root == 'LEVIR':
        args.train_data_root = '../BCD/LEVIR+512'
        args.test_data_root_1 = '../BCD/LEVIR+512'
        args.test_data_root_2 = '../BCD/BCDD-512'
    elif args.file_root == 'GVLM':
        args.train_data_root = '../BCD/GVLM-512'
        args.test_data_root_1 = '../BCD/GVLM-512'
        args.test_data_root_2 = '../BCD/GVLM-512'
    elif args.file_root == 'MCD':
        args.train_data_root = '../BCD/MCD-512'
        args.test_data_root_1 = '../BCD/MCD-512'
        args.test_data_root_2 = '../BCD/MCD-512'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])
    train_data = myDataLoader.Dataset("train", file_root=args.train_data_root, transform=trainDataset_main)
    trainLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )
    val_data = myDataLoader.Dataset("val", file_root=args.train_data_root, transform=valDataset)
    valLoader = torch.utils.data.DataLoader(
        val_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)
    test_data_1 = myDataLoader.Dataset("test", file_root=args.test_data_root_1, transform=valDataset)
    testLoader_1 = torch.utils.data.DataLoader(
        test_data_1, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)
    test_data_2 = myDataLoader.Dataset("test", file_root=args.test_data_root_2, transform=valDataset)
    testLoader_2 = torch.utils.data.DataLoader(
        test_data_2, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    max_batches = len(trainLoader)
    print('For each epoch, we have {} batches'.format(max_batches))

    if args.onGPU:
        cudnn.benchmark = True

    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0
    max_F1_val = 0

    logFileLoc = args.save_dir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa (val)', 'IoU (val)', 'F1 (val)', 'R (val)', 'P (val)'))
    logger.flush()

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, (0.9, 0.999), weight_decay=1e-2)
    model_file_name = args.save_dir + 'best_model.pth'
    criterion = DeepSupervisionLoss()
    for epoch in range(start_epoch, args.max_epochs):
        lossTr, score_tr, lr = \
            train(args, trainLoader, model, criterion, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader)
        torch.cuda.empty_cache()
        # evaluate on validation set
        if epoch == 0:
            continue
        lossVal, score_val = val(args, valLoader, model, criterion)
        torch.cuda.empty_cache()
        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (epoch, score_val['Kappa'], score_val['IoU'],
                                                                       score_val['F1'], score_val['recall'],
                                                                       score_val['precision']))
        logger.flush()
        # save the model
        if epoch % 1 == 0 and max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']
            torch.save(model.state_dict(), model_file_name)
        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t F1(tr) = %.4f\t F1(val) = %.4f" \
              % (epoch, lossTr, lossVal, score_tr['F1'], score_val['F1'])
              )
    #
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)
    #
    loss_test_set1, score_test_set1 = val(args, testLoader_1, model, criterion)
    torch.cuda.empty_cache()
    print("\nSet1_Test :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test_set1['Kappa'], score_test_set1['IoU'], score_test_set1['F1'], score_test_set1['recall'],
             score_test_set1['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('Set1_Test',
                                                                   score_test_set1['Kappa'],
                                                                   score_test_set1['IoU'],
                                                                   score_test_set1['F1'],
                                                                   score_test_set1['recall'],
                                                                   score_test_set1['precision']))
    logger.flush()
    #
    loss_test_set2, score_test_set2 = val(args, testLoader_2, model, criterion)
    torch.cuda.empty_cache()
    print("\nSet2_Test :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test_set2['Kappa'], score_test_set2['IoU'], score_test_set2['F1'], score_test_set2['recall'],
             score_test_set2['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('Set2_Test',
                                                                   score_test_set2['Kappa'],
                                                                   score_test_set2['IoU'],
                                                                   score_test_set2['F1'],
                                                                   score_test_set2['recall'],
                                                                   score_test_set2['precision']))
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=4079, help='Seed')
    parser.add_argument('--model_name', default="HRMNet", help='Data directory')
    parser.add_argument('--file_root', default="MCD", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=512, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy')
    parser.add_argument('--save_dir', default='./weights/', help='Directory to save the results')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    train_val_change_detection(args)
