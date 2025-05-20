from timm.utils import accuracy, AverageMeter

import logging
import time
import torch
import torch.distributed as dist
import math

from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed
from .task_name import ANALYSIS_TASKS

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

class LimitedAvgMeter(object):
    def __init__(self, max_num=10, best_mode="max"):
        self.avg = 0.0
        self.num_list = []
        self.max_num = max_num
        self.best_mode = best_mode
        self.best = 0.0 if best_mode == "max" else 100.0

    def append(self, x):
        self.num_list.append(x)
        len_list = len(self.num_list)
        if len_list > 0:
            if len_list < self.max_num:
                self.avg = sum(self.num_list) / len_list
            else:
                self.avg = sum(self.num_list[(len_list - self.max_num):len_list]) / self.max_num

        if self.best_mode == "max":
            if self.avg > self.best:
                self.best = self.avg
        elif self.best_mode == "min":
            if self.avg < self.best:
                self.best = self.avg

class AttributeVerification(object):
    def __init__(self, data_loader, summary_writer=None):
        self.rank: int = distributed.get_rank()
        self.data_loader = data_loader
        self.summary_writer = summary_writer

        # Gender accuracy metrics
        self.gender_meter = LimitedAvgMeter(best_mode="max")
        self.highest_gender_acc = 0.0

        # Age error metrics
        self.age_meter = LimitedAvgMeter(best_mode="min")
        self.best_age_mae = 100.0

        # Height error metrics
        self.height_meter = LimitedAvgMeter(best_mode="min")
        self.best_height_mae = 100.0

        # Weight error metrics
        self.weight_meter = LimitedAvgMeter(best_mode="min")
        self.best_weight_mae = 100.0

        # BMI error metrics
        self.bmi_meter = LimitedAvgMeter(best_mode="min")
        self.best_bmi_mae = 100.0

    def ver_test(self, model, gender_subnet, age_subnet, height_subnet, weight_subnet, bmi_subnet, global_step):
        logging.info("Val on Attributes:")

        # Initialize loss functions
        gender_criterion = torch.nn.CrossEntropyLoss()
        age_criterion = torch.nn.MSELoss()
        height_criterion = torch.nn.MSELoss()
        weight_criterion = torch.nn.MSELoss()
        bmi_criterion = torch.nn.MSELoss()

        # Initialize meters
        gender_loss_meter = AverageMeter()
        gender_acc_meter = AverageMeter()
        age_loss_meter = AverageMeter()
        age_error_meter = AverageMeter()
        height_loss_meter = AverageMeter()
        height_error_meter = AverageMeter()
        weight_loss_meter = AverageMeter()
        weight_error_meter = AverageMeter()
        bmi_loss_meter = AverageMeter()
        bmi_error_meter = AverageMeter()

        batch_time = AverageMeter()

        end = time.time()
        for idx, (images, [gender_target, age_target, height_target, weight_target, bmi_target]) in enumerate(self.data_loader):
            img = images.cuda(non_blocking=True)
            gender_target = gender_target.cuda(non_blocking=True)
            age_target = age_target.cuda(non_blocking=True)
            height_target = height_target.cuda(non_blocking=True)
            weight_target = weight_target.cuda(non_blocking=True)
            bmi_target = bmi_target.cuda(non_blocking=True)

            # Get features
            _, global_features, _ = model.module.backbone.forward_features(img)

            # Compute outputs
            [gender_output] = gender_subnet(global_features)
            [age_output] = age_subnet(global_features)
            [height_output] = height_subnet(global_features)
            [weight_output] = weight_subnet(global_features)
            [bmi_output] = bmi_subnet(global_features)

            # Compute losses and metrics
            # Gender
            gender_loss = gender_criterion(gender_output, gender_target)
            gender_acc, _ = accuracy(gender_output, gender_target, topk=(1, 1))
            gender_loss = reduce_tensor(gender_loss)
            gender_acc = reduce_tensor(gender_acc)
            gender_loss_meter.update(gender_loss.item(), gender_target.size(0))
            gender_acc_meter.update(gender_acc.item(), gender_target.size(0))

            # Age
            age_loss = age_criterion(age_output, age_target)
            age_error = torch.mean(torch.abs(age_output - age_target))
            age_loss = reduce_tensor(age_loss)
            age_error = reduce_tensor(age_error)
            age_loss_meter.update(age_loss.item(), age_target.size(0))
            age_error_meter.update(age_error.item(), age_target.size(0))

            # Height
            height_loss = height_criterion(height_output, height_target)
            height_error = torch.mean(torch.abs(height_output - height_target))
            height_loss = reduce_tensor(height_loss)
            height_error = reduce_tensor(height_error)
            height_loss_meter.update(height_loss.item(), height_target.size(0))
            height_error_meter.update(height_error.item(), height_target.size(0))

            # Weight
            weight_loss = weight_criterion(weight_output, weight_target)
            weight_error = torch.mean(torch.abs(weight_output - weight_target))
            weight_loss = reduce_tensor(weight_loss)
            weight_error = reduce_tensor(weight_error)
            weight_loss_meter.update(weight_loss.item(), weight_target.size(0))
            weight_error_meter.update(weight_error.item(), weight_target.size(0))

            # BMI
            bmi_loss = bmi_criterion(bmi_output, bmi_target)
            bmi_error = torch.mean(torch.abs(bmi_output - bmi_target))
            bmi_loss = reduce_tensor(bmi_loss)
            bmi_error = reduce_tensor(bmi_error)
            bmi_loss_meter.update(bmi_loss.item(), bmi_target.size(0))
            bmi_error_meter.update(bmi_error.item(), bmi_target.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info(
                    f'Test: [{idx}/{len(self.data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        # Update best metrics
        if gender_acc_meter.avg > self.highest_gender_acc:
            self.highest_gender_acc = gender_acc_meter.avg
        if age_error_meter.avg < self.best_age_mae:
            self.best_age_mae = age_error_meter.avg
        if height_error_meter.avg < self.best_height_mae:
            self.best_height_mae = height_error_meter.avg
        if weight_error_meter.avg < self.best_weight_mae:
            self.best_weight_mae = weight_error_meter.avg
        if bmi_error_meter.avg < self.best_bmi_mae:
            self.best_bmi_mae = bmi_error_meter.avg

        # Update limited meters
        self.gender_meter.append(gender_acc_meter.avg)
        self.age_meter.append(age_error_meter.avg)
        self.height_meter.append(height_error_meter.avg)
        self.weight_meter.append(weight_error_meter.avg)
        self.bmi_meter.append(bmi_error_meter.avg)

        if self.rank == 0:
            self.summary_writer: SummaryWriter

            # Log metrics
            # Gender
            self.summary_writer.add_scalar('Gender Val Loss', gender_loss_meter.avg, global_step)
            self.summary_writer.add_scalar('Gender Val Accuracy', gender_acc_meter.avg, global_step)
            logging.info(f'[{global_step}]Gender Loss: {gender_loss_meter.avg:.5f}')
            logging.info(f'[{global_step}]Gender Accuracy: {gender_acc_meter.avg:.5f}')
            logging.info(f'[{global_step}]Gender Highest Accuracy: {self.highest_gender_acc:.5f}')
            logging.info(f'[{global_step}]Gender 10 Times Accuracy: {self.gender_meter.avg:.5f}')
            logging.info(f'[{global_step}]Gender 10 Times Best Accuracy: {self.gender_meter.best:.5f}')

            # Age
            self.summary_writer.add_scalar('Age Val Loss', age_loss_meter.avg, global_step)
            self.summary_writer.add_scalar('Age Val MAE', age_error_meter.avg, global_step)
            logging.info(f'[{global_step}]Age Loss: {age_loss_meter.avg:.5f}')
            logging.info(f'[{global_step}]Age MAE: {age_error_meter.avg:.5f}')
            logging.info(f'[{global_step}]Age Best MAE: {self.best_age_mae:.5f}')
            logging.info(f'[{global_step}]Age 10 Times MAE: {self.age_meter.avg:.5f}')
            logging.info(f'[{global_step}]Age 10 Times Best MAE: {self.age_meter.best:.5f}')

            # Height
            self.summary_writer.add_scalar('Height Val Loss', height_loss_meter.avg, global_step)
            self.summary_writer.add_scalar('Height Val MAE', height_error_meter.avg, global_step)
            logging.info(f'[{global_step}]Height Loss: {height_loss_meter.avg:.5f}')
            logging.info(f'[{global_step}]Height MAE: {height_error_meter.avg:.5f}')
            logging.info(f'[{global_step}]Height Best MAE: {self.best_height_mae:.5f}')
            logging.info(f'[{global_step}]Height 10 Times MAE: {self.height_meter.avg:.5f}')
            logging.info(f'[{global_step}]Height 10 Times Best MAE: {self.height_meter.best:.5f}')

            # Weight
            self.summary_writer.add_scalar('Weight Val Loss', weight_loss_meter.avg, global_step)
            self.summary_writer.add_scalar('Weight Val MAE', weight_error_meter.avg, global_step)
            logging.info(f'[{global_step}]Weight Loss: {weight_loss_meter.avg:.5f}')
            logging.info(f'[{global_step}]Weight MAE: {weight_error_meter.avg:.5f}')
            logging.info(f'[{global_step}]Weight Best MAE: {self.best_weight_mae:.5f}')
            logging.info(f'[{global_step}]Weight 10 Times MAE: {self.weight_meter.avg:.5f}')
            logging.info(f'[{global_step}]Weight 10 Times Best MAE: {self.weight_meter.best:.5f}')

            # BMI
            self.summary_writer.add_scalar('BMI Val Loss', bmi_loss_meter.avg, global_step)
            self.summary_writer.add_scalar('BMI Val MAE', bmi_error_meter.avg, global_step)
            logging.info(f'[{global_step}]BMI Loss: {bmi_loss_meter.avg:.5f}')
            logging.info(f'[{global_step}]BMI MAE: {bmi_error_meter.avg:.5f}')
            logging.info(f'[{global_step}]BMI Best MAE: {self.best_bmi_mae:.5f}')
            logging.info(f'[{global_step}]BMI 10 Times MAE: {self.bmi_meter.avg:.5f}')
            logging.info(f'[{global_step}]BMI 10 Times Best MAE: {self.bmi_meter.best:.5f}')

    def __call__(self, num_update, model, gender_subnet, age_subnet, height_subnet, weight_subnet, bmi_subnet):
        model.eval()
        self.ver_test(model, gender_subnet, age_subnet, height_subnet, weight_subnet, bmi_subnet, num_update)
        model.train()


