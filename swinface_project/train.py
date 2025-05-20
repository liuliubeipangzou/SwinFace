import argparse
import logging
import os
from itertools import cycle
import math

from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lr_scheduler import build_scheduler
from analysis import *
from analysis import subnets
from analysis.losses import AgeLoss, HeightLoss, WeightLoss, BMILoss
from model import build_model

from utils.utils_callbacks import CallBackLogging
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    attribute_train_loader = get_analysis_train_dataloader("attribute", cfg, args.local_rank)
    # Analysis dataloaders
    model = build_model(cfg).cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    model.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    model._set_static_graph()

    cfg.total_batch_size = world_size * cfg.attribute_bz
    cfg.epoch_step = len(attribute_train_loader)

    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    age_loss = AgeLoss(total_iter=cfg.total_step)

    # Analysis task_losses
    criteria = [
        torch.nn.CrossEntropyLoss(),  # Gender
        age_loss,  # Age
        HeightLoss(),  # Height
        WeightLoss(),  # Weight
        BMILoss(),  # BMI
    ]  # Total:5

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
                    ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
                    ],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    lr_scheduler = build_scheduler(
        optimizer=opt,
        lr_name=cfg.lr_name,
        warmup_lr=cfg.warmup_lr,
        min_lr=cfg.min_lr,
        num_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step)

    start_epoch = 0
    global_step = 0

    if cfg.init:
        dict_checkpoint = torch.load(os.path.join(cfg.init_model, f"start_{rank}.pt"))
        model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"],
                                              strict=False)  # only load backbone!
        del dict_checkpoint

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_step_{cfg.resume_step}_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        local_step = dict_checkpoint["local_step"]

        if local_step == cfg.epoch_step - 1:
            start_epoch = start_epoch+1
            local_step = 0
        else:
            local_step += 1

        global_step += 1

        model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        model.module.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
        model.module.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
        model.module.om.load_state_dict(dict_checkpoint["state_dict_om"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))



    attribute_val_loader = get_analysis_val_dataloader(data_choose="attribute", config=cfg)
    attribute_verification = AttributeVerification(data_loader=attribute_val_loader, summary_writer=summary_writer)

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.attribute_bz,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    analysis_loss_ams = [AverageMeter() for j in range(5)] # Gender, Age, Height, Weight, BMI

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    bzs = [cfg.attribute_bz]

    features_cut = [0 for i in range(2)]
    for i in range(1, 2):
        features_cut[i] = features_cut[i - 1] + bzs[i - 1]



    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(attribute_train_loader, DataLoader):
            attribute_train_loader.sampler.set_epoch(epoch)

        for idx, attribute in enumerate(attribute_train_loader):

            # skip
            if cfg.resume:
                if idx < local_step:
                    continue

            attribute_img, [gender_label, age_label, height_label, weight_label, bmi_label] = attribute

            gender_label = gender_label.cuda(non_blocking=True)
            age_label = age_label.cuda(non_blocking=True)
            height_label = height_label.cuda(non_blocking=True)
            weight_label = weight_label.cuda(non_blocking=True)
            bmi_label = bmi_label.cuda(non_blocking=True)

            analysis_labels = [gender_label, age_label, height_label, weight_label, bmi_label] # Gender, Age, Height, Weight, BMI

            img = attribute_img.cuda(non_blocking=True)

            model.module.set_output_type("List")
            outputs = model(img)

            analysis_losses = []

            # Assuming the order of outputs corresponds to [Gender, Age, Height, Weight, BMI]
            # All analysis outputs now come from the attribute batch
            gender_output = outputs[0][features_cut[0]: features_cut[1]]
            age_output = outputs[1][features_cut[0]: features_cut[1]]
            height_output = outputs[2][features_cut[0]: features_cut[1]]
            weight_output = outputs[3][features_cut[0]: features_cut[1]]
            bmi_output = outputs[4][features_cut[0]: features_cut[1]]

            analysis_outputs = [gender_output, age_output, height_output, weight_output, bmi_output]

            loss = 0
            for j in range(5):
                if j == 1:  # age
                    analysis_loss = criteria[j](analysis_outputs[j], analysis_labels[j], global_step)
                else:
                    analysis_loss = criteria[j](analysis_outputs[j], analysis_labels[j])
                analysis_losses.append(analysis_loss)
                loss += analysis_losses[j] * cfg.analysis_loss_weights[j]

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step_update(global_step)

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                for j in range(5):
                    analysis_loss_ams[j].update(analysis_losses[j].item(), 1)

                callback_logging(global_step, loss_am, None, analysis_loss_ams, epoch, cfg.fp16,
                                 lr_scheduler.get_update_values(global_step)[0], amp)

                if (global_step+1) % cfg.verbose == 0:
                    model.module.set_output_type("Attribute")
                    attribute_verification(global_step, model, model.module.fam, model.module.tss, model.module.om, model.module.om, model.module.om)

            if cfg.save_all_states and (global_step+1) % cfg.save_verbose == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "local_step": idx,

                    "state_dict_backbone": model.module.backbone.state_dict(),
                    "state_dict_fam": model.module.fam.state_dict(), # Assuming fam is for Gender
                    "state_dict_tss": model.module.tss.state_dict(), # Assuming tss is for Age
                    "state_dict_om": model.module.om.state_dict(), # Assuming om is for Attribute
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_step_{global_step}_gpu_{rank}.pt"))

            # update
            if global_step >= cfg.total_step - 1:
                break  # end
            else:
                global_step += 1

        if global_step >= cfg.total_step - 1:
            break


    with torch.no_grad():
        model.module.set_output_type("Attribute")
        attribute_verification(global_step, model, model.module.fam, model.module.tss, model.module.om, model.module.om, model.module.om)

    # if rank == 0:
    # path_module = os.path.join(cfg.output, "model.pt")
    # torch.save(backbone.module.state_dict(), path_module)

    # from torch2onnx import convert_onnx
    # convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
