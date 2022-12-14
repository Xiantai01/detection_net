import time
import os
import datetime

import torch

import transforms
from my_dataset import VOCDataSet
from backbone import resnet50_fpn_model
from network_files import FasterRCNN, FastRCNNPredictor
import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir


def create_model(num_classes):

    backbone = resnet50_fpn_model(norm_layer=torch.nn.BatchNorm2d,
                                  trainable_layers=3)

    model = FasterRCNN(backbone=backbone, num_classes=91)
    weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)


    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    print("Loading data")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path

    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")


    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
   
    model = create_model(num_classes=args.num_classes + 1)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')  
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        utils.evaluate(model, data_loader_test, device=device)
        return

    train_loss = []
    learning_rate = []
    val_map = []

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader,
                                              device, epoch, args.print_freq,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update learning rate
        lr_scheduler.step()

        # evaluate after every epoch
        coco_info = utils.evaluate(model, data_loader_test, device=device)
        val_map.append(coco_info[1])  # pascal mAP

        
        if args.rank in [-1, 0]:
            # write into txt
            with open(results_file, "a") as f:
               
                result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

        if args.output_dir:
            
            save_files = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files,
                           os.path.join(args.output_dir, f'model_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.rank in [-1, 0]:
        # plot loss and lr curve
        if len(train_loss) != 0 and len(learning_rate) != 0:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate)

        # plot mAP curve
        if len(val_map) != 0:
            from plot_curve import plot_map
            plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

   
    parser.add_argument('--data-path', default='./', help='dataset')
   
    parser.add_argument('--device', default='cuda', help='device')
 
    parser.add_argument('--num-classes', default=20, type=int, help='num_classes')
    
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
  
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
   
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
   
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
  
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
   
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
   
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
  
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
  
    parser.add_argument('--lr-steps', default=[7, 12], nargs='+', type=int, help='decrease lr every step-size epochs')
   
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
   
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

   
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
