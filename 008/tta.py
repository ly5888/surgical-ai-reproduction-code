import time
import os
import torch
import os.path as osp
import numpy as np
import logging
from functools import reduce
from options.train_options import TrainOptions
from options.test_options import TestOptions
from loaders import create_dataset
from models import create_model
from util.logger import define_logger
from metrics import cal_confusion_for_one_img, summarise_statistics, cal_iou, cal_dice
from PIL import Image


class Runner():
    def __init__(self):
        self.gather_options()
        self.environment_settings()
        define_logger(self.save_dir, 'tta')
        self.logger = logging.getLogger('tta')
        self.train_parser.print_options(self.opt, time.time())
        self.dataset_preparation()
        self.model_preparation()
        self.run()

    def environment_settings(self):
        # create directory
        os.makedirs(osp.join(self.opt.checkpoints_dir, self.opt.name), exist_ok=True)
        self.logger = logging.getLogger('tta')

        self.opt.checkpoints_dir = osp.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(osp.join(self.opt.checkpoints_dir, self.opt.name), exist_ok=True)
        torch.backends.cudnn.benchmark = True
        self.save_name = self.opt.name + ('_' + self.opt.save_suffix if self.opt.save_suffix != "" else '')
        self.save_dir = osp.join(self.opt.results_dir, self.save_name)
        os.makedirs(osp.join(self.save_dir, 'image'), exist_ok=True)

        # attributes initializeation

    def gather_options(self):
        self.train_parser = TrainOptions()
        train_opt = self.train_parser.parse()
        self.opt = train_opt

    def dataset_preparation(self):
        self.opt.phase = 'test'
        self.dataset = create_dataset(self.opt, self.logger)
        self.opt.total_batches = len(self.dataset.dataloader)
        self.logger.info("===================================================================")
        self.logger.info("number of tta samples: %d, number of batches per epoch %d" % (len(self.dataset), len(self.dataset.dataloader)))
        self.logger.info("===================================================================")

    def model_preparation(self):
        self.opt.phase = 'training'
        self.model = create_model(self.opt, self.logger)
        # set logger 
        self.model.initialize()
        self.model.print_infos()

    def save_visuals(self, visuals, idx):
        for name, image in visuals.items():
            if name != 'rgb' and name != 'color_pred':
                continue
            # convert to PIL Image
            if image.ndim == 3:
                image = image.transpose((1, 2, 0)).squeeze()   
            image = image.astype(np.uint8)         
            image = Image.fromarray(image)

            # save to results dir
            path = osp.join(self.save_dir, 'image', "{}_{}.png".format(idx, name))
            image.save(path)
            
    def run(self):
        """对每张测试图像独立执行MUTA适配，并统计三分类全局指标。"""
        from copy import deepcopy

        # 保存适配前的源模型与优化器状态。
        initial_network_states = {
            name: deepcopy(
                getattr(self.model, "net" + name).state_dict()
            )
            for name in self.model.net_names
        }
        initial_optimizer_states = [
            deepcopy(optimizer.state_dict())
            for optimizer in self.model.optimizers
        ]

        total_tp = np.zeros(self.opt.output_nc, dtype=np.float64)
        total_fp = np.zeros(self.opt.output_nc, dtype=np.float64)
        total_fn = np.zeros(self.opt.output_nc, dtype=np.float64)

        for idx, data in enumerate(self.dataset):
            # 每张图像都从完全相同的源模型开始，避免跨样本累积适配。
            for name, state in initial_network_states.items():
                getattr(self.model, "net" + name).load_state_dict(
                    deepcopy(state)
                )

            for optimizer, state in zip(
                self.model.optimizers,
                initial_optimizer_states
            ):
                optimizer.load_state_dict(deepcopy(state))
                optimizer.zero_grad(set_to_none=True)

            self.model.n_iters = 0
            self.model.train()

            # 对当前测试样本独立执行指定次数的适配。
            for step in range(self.opt.tta_steps):
                self.model.n_iters = step
                self.model.set_inputs(data)
                self.model.optimize()

            # 此时仍保留最后一次适配产生的伪标签，可正常保存可视化。
            visuals = self.model.get_visuals(batch_index=0)
            self.save_visuals(visuals, idx)

            # 使用当前样本适配后的学生模型生成最终评估预测。
            self.model.eval()
            self.model.set_inputs(data)
            with torch.no_grad():
                self.model.forward()
                self.model.visualization_preprocess()

            mask = self.model.mask.cpu().numpy().astype(np.uint32)
            gt = self.model.gt.cpu().numpy().astype(np.uint32)

            tp, fp, fn = cal_confusion_for_one_img(
                self.opt.output_nc, gt, mask
            )
            total_tp += np.asarray(tp, dtype=np.float64)
            total_fp += np.asarray(fp, dtype=np.float64)
            total_fn += np.asarray(fn, dtype=np.float64)

            sample_iou = cal_iou(tp, fp, fn)
            sample_dice = cal_dice(tp, fp, fn)
            self.logger.info(
                f"{idx + 1}/{len(self.dataset)} {data['name']}: "
                f"mIoU={np.nanmean(sample_iou):.3f}, "
                f"mDice={np.nanmean(sample_dice):.3f}"
            )

        ious = np.asarray(
            cal_iou(total_tp, total_fp, total_fn),
            dtype=np.float64
        )
        dices = np.asarray(
            cal_dice(total_tp, total_fp, total_fn),
            dtype=np.float64
        )

        class_names = [
            "Background",
            "Surgical Tools",
            "Eye Retractor",
        ]

        self.logger.info("=" * 72)
        for class_id, (iou, dice) in enumerate(zip(ious, dices)):
            name = (
                class_names[class_id]
                if class_id < len(class_names)
                else f"Class {class_id}"
            )
            self.logger.info(
                f"{name:<18} IoU={iou:8.3f}%  Dice={dice:8.3f}%"
            )

        self.logger.info("-" * 72)
        self.logger.info(
            f"mIoU（三类）: {np.nanmean(ious):.3f}%"
        )
        self.logger.info(
            f"mDice（三类）: {np.nanmean(dices):.3f}%"
        )
        self.logger.info(
            f"前景mIoU: {np.nanmean(ious[1:]):.3f}%"
        )
        self.logger.info(
            f"前景mDice: {np.nanmean(dices[1:]):.3f}%"
        )
        self.logger.info("=" * 72)


if __name__ == '__main__':
    runner = Runner()
