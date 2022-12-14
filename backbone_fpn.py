import os
import datetime

import torch
from Xiantai_dataset import NEU_DataSet
import transforms
from network_files import FasterRCNN, AnchorsGenerator
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from backbone import BackboneWithFPN, LastLevelMaxPool
from backbone.model_hrnet import LiteHRNet
from backbone.Simsiam import SimSiam

def create_model(num_classes):
    import torchvision
    base_channel = 16
    extra = dict(
        stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=(2, 4, 2),
            num_branches=(2, 3, 4),
            num_blocks=(2, 2, 2),
            module_type=('LITE', 'LITE', 'LITE'),
            with_fuse=(True, True, True),
            reduce_ratios=(8, 8, 8),
            num_channels=(
                (base_channel, base_channel * 2),
                (base_channel, base_channel * 2, base_channel * 4),
                (base_channel, base_channel * 2, base_channel * 4, base_channel * 8),
            )),
    )
    new_backbone = LiteHRNet(extra, include_top=False, in_channels=3)
    checkpoint = torch.load('./backbone/ckpt_epoch_30_2_litehrnet.pth')
    model_pre = SimSiam()
    model_pre.load_state_dict(checkpoint['state_dict'])
    new_dict = new_backbone.state_dict()
    pretext_model = model_pre.backbone
    # state_dict = {k: v for k, v in pretext_model.state_dict() if k in new_dict.keys()}
    # new_dict.update(state_dict)
    new_backbone.load_state_dict(pretext_model.state_dict())
    # img = torch.rand(1,3,256,256)
    # print(model_HR(img).shape)

    # --- mobilenet_v3_large fpn backbone --- #
    # backbone = torchvision.models.mobilenet_v3_large(pretrained=False)
    # print(backbone)
    # return_layers = {"features.6": "0",   # stride 8
    #                  "features.12": "1",  # stride 16
    #                  "features.16": "2"}  # stride 32
    in_channels_list = [16, 32, 64, 128]
    # new_backbone = create_feature_extractor(backbone, return_layers)
    # img = torch.randn(2, 3, 224, 224)
    # outputs = new_backbone(img)
    # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]

    # --- efficientnet_b0 fpn backbone --- #
    # backbone = torchvision.models.efficientnet_b0(pretrained=True)
    # # print(backbone)
    # return_layers = {"features.3": "0",  # stride 8
    #                  "features.4": "1",  # stride 16
    #                  "features.8": "2"}  # stride 32
    # in_channels_list = [40, 80, 1280]
    # new_backbone = create_feature_extractor(backbone, return_layers)
    # # img = torch.randn(1, 3, 224, 224)
    # # outputs = new_backbone(img)
    # # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]

    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=None,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)

    anchor_sizes = ((16,), (32,), (64,), (128,), (256,),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],  
                                                    output_size=[7, 7],  
                                                    sampling_ratio=2) 

    model = FasterRCNN(backbone=backbone_with_fpn,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def main(args):
    # model1 = SimSiam()
    # model1 = model1.backbone
    # base_channel = 16
    # extra = dict(
    #     stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
    #     num_stages=3,
    #     stages_spec=dict(
    #         num_modules=(2, 4, 2),
    #         num_branches=(2, 3, 4),
    #         num_blocks=(2, 2, 2),
    #         module_type=('LITE', 'LITE', 'LITE'),
    #         with_fuse=(True, True, True),
    #         reduce_ratios=(8, 8, 8),
    #         num_channels=(
    #             (base_channel, base_channel * 2),
    #             (base_channel, base_channel * 2, base_channel * 4),
    #             (base_channel, base_channel * 2, base_channel * 4, base_channel * 8),
    #         )),
    # )
    # # print(extra['stages_spec']['num_channels'][0][0])
    # model2 = LiteHRNet(extra, include_top=False, in_channels=3)
    # dict1 = model1.state_dict()
    # dict2 = model2.state_dict()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    data_root = args.data_path
  
    train_dataset = NEU_DataSet(data_root, data_transform["train"], "train.txt")
    train_sampler = None

    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
      
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
      
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

   
    batch_size = args.batch_size
    nw = 0  # number of workers
    # print('Using %g dataloader workers' % nw)
    if train_sampler:
      
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    val_dataset = NEU_DataSet(data_root, data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=args.num_classes + 1)
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params,
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10,
                                                   gamma=0.33)

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

       
        coco_info = utils.evaluate(model, val_data_set_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
           
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])

        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "./save_weights/LiteHRNet-model-{}.pth".format(epoch))

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

 
    parser.add_argument('--device', default='cuda:0', help='device')
  
    parser.add_argument('--data-path', default='./NEU-DEF', help='dataset')
  
    parser.add_argument('--num-classes', default=6, type=int, help='num_classes')
  
    parser.add_argument('--output-dir', default='./save_weights_light', help='path where to save')
   
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
   
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
 
    parser.add_argument('--epochs', default=35, type=int, metavar='N',
                        help='number of total epochs to run')
 
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
   
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
   
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    args.batch_size = 16
    args.epochs = 10
    args.lr = 0.2
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
