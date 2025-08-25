from .nets.yolo import YoloBody
from .nets.yolo_training import YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init
from .utils.callbacks import EvalCallback, LossHistory
from .utils.dataloader import YoloDataset, yolo_dataset_collate
from .utils.utils import get_anchors, get_classes, seed_everything, show_config, worker_init_fn
from .utils.utils_fit import fit_one_epoch
import os, datetime, torch, numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial


def train_yolov4(
    Cuda=True,
    seed=11,
    distributed=False,
    sync_bn=False,
    fp16=False,
    classes_path='model_data/voc_classes.txt',
    anchors_path='model_data/yolo_anchors.txt',
    anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    model_path='model_data/yolo4_weights.pth',
    input_shape=[416, 416],
    pretrained=False,
    mosaic=True,
    mosaic_prob=0.5,
    mixup=True,
    mixup_prob=0.5,
    special_aug_ratio=0.7,
    label_smoothing=0,
    Init_Epoch=0,
    Freeze_Epoch=50,
    Freeze_batch_size=8,
    UnFreeze_Epoch=300,
    Unfreeze_batch_size=4,
    Freeze_Train=True,
    Init_lr=1e-2,
    optimizer_type="sgd",
    momentum=0.937,
    weight_decay=5e-4,
    lr_decay_type="cos",
    focal_loss=False,
    focal_alpha=0.25,
    focal_gamma=2,
    iou_type='ciou',
    save_period=10,
    save_dir='logs',
    eval_flag=True,
    eval_period=1,
    num_workers=4,
    train_annotation_path='2007_train.txt',
    val_annotation_path='2007_val.txt'
):
    

    seed_everything(seed)

    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    model = YoloBody(anchors_mask, num_classes, pretrained=pretrained)

    if not pretrained:
        weights_init(model)

    if model_path:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        temp_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing, focal_loss, focal_alpha, focal_gamma, iou_type)
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape) if local_rank == 0 else None
    scaler = torch.cuda.amp.GradScaler() if fp16 else None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type in ['adam', 'adamw'] else 5e-2
    lr_limit_min = 3e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * (Init_lr * 0.01), lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optimizer = {
        'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
        'adamw': optim.AdamW(pg0, Init_lr_fit, betas=(momentum, 0.999)),
        'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
    }[optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    train_dataset = YoloDataset(train_lines, input_shape, num_classes, UnFreeze_Epoch, mosaic, mixup, mosaic_prob, mixup_prob, True, special_aug_ratio)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, UnFreeze_Epoch, False, False, 0, 0, False, 0)

    shuffle = not distributed
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if distributed else None
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, Cuda, eval_flag=eval_flag, period=eval_period) if local_rank == 0 else None

    UnFreeze_flag = False
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            for param in model.backbone.parameters():
                param.requires_grad = True
            UnFreeze_flag = True

        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch
        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()

