# -*- coding: UTF-8 -*-
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models.base_model import BaseModel
from models.networks import define_network


class MUTAPretrainModel(BaseModel):
    """MUTA多视图源域监督预训练模型。"""

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):
        if isTrain:
            parser.add_argument("--lambda_con", type=float, default=0.05)
            parser.add_argument("--lambda_dis", type=float, default=0.3)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=torch.cuda.is_available()
        )

    def register_nets(self):
        # 保存时对应 final_net_E/D1/D2.pth，
        # 与现有muta_source_model.py的加载逻辑兼容。
        self.net_names = ["E", "D1", "D2"]

        self.netE = define_network(
            self.opt.input_nc,
            self.opt.output_nc,
            "res50_encoder",
            self.opt.gpu,
            "in_model",
            self.opt.init_gain,
            {"pretrained": True},
        )

        self.netD1 = define_network(
            self.opt.input_nc,
            self.opt.output_nc,
            "aspp",
            self.opt.gpu,
            "normal",
            self.opt.init_gain,
        )

        self.netD2 = define_network(
            self.opt.input_nc,
            self.opt.output_nc,
            "aspp",
            self.opt.gpu,
            "normal",
            self.opt.init_gain,
        )

    def register_optimizers(self):
        parameters = list(self.netE.parameters())
        parameters += list(self.netD1.parameters())
        parameters += list(self.netD2.parameters())

        self.optimizer_MUTA = torch.optim.Adam(
            parameters,
            lr=self.opt.lr,
        )

    def register_losses(self):
        self.loss_names = [
            "seg1",
            "seg2",
            "con",
            "dis",
            "total",
        ]

        self.criterion_seg = torch.nn.CrossEntropyLoss(
            ignore_index=self.opt.ignore_label
        )

    def register_visuals(self):
        self.visual_names = [
            "rgb",
            "color_gt",
            "color_pred",
        ]

    def set_inputs(self, data):
        self.rgb = data["image"].to(
            self.device,
            non_blocking=True,
        )
        self.gt = data["label"].to(
            self.device,
            dtype=torch.long,
            non_blocking=True,
        )

    def photometric_perturbation(self, images):
        # 仓库加载器把图像归一化到[-1,1]；
        # 先还原到[0,1]进行亮度、对比度、颜色和模糊扰动。
        images = (images * 0.5 + 0.5).clamp(0, 1)
        outputs = []

        for image in images:
            image = TF.adjust_brightness(
                image,
                random.uniform(0.75, 1.25),
            )
            image = TF.adjust_contrast(
                image,
                random.uniform(0.75, 1.25),
            )
            image = TF.adjust_saturation(
                image,
                random.uniform(0.75, 1.25),
            )

            if random.random() < 0.5:
                image = TF.gaussian_blur(
                    image,
                    kernel_size=[5, 5],
                    sigma=[0.1, 1.5],
                )

            image = image.clamp(0, 1)
            image = (image - 0.5) / 0.5
            outputs.append(image)

        return torch.stack(outputs, dim=0)

    @staticmethod
    def decode(decoder, feature, output_size):
        prediction = decoder(feature)
        return F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )

    def decoder_dissimilarity(self):
        # 论文公式(2)：cos(w1,w2)+1，并在线性训练过程中衰减。
        dot = torch.zeros((), device=self.device)
        norm1 = torch.zeros((), device=self.device)
        norm2 = torch.zeros((), device=self.device)

        for parameter1, parameter2 in zip(
            self.netD1.parameters(),
            self.netD2.parameters(),
        ):
            dot = dot + torch.sum(parameter1 * parameter2)
            norm1 = norm1 + torch.sum(parameter1.square())
            norm2 = norm2 + torch.sum(parameter2.square())

        cosine = dot / (
            torch.sqrt(norm1 * norm2) + 1e-8
        )

        decay = max(
            0.0,
            1.0 - float(self.curr_epoch) /
            max(1, self.opt.epoch_end),
        )

        return (cosine + 1.0) * decay

    def forward(self):
        output_size = self.rgb.shape[-2:]
        feature = self.netE(self.rgb)

        if self.isTrain:
            perturbed_rgb = self.photometric_perturbation(
                self.rgb
            )
            perturbed_feature = self.netE(perturbed_rgb)

            self.feature = feature
            self.perturbed_feature = perturbed_feature

            self.pred1 = self.decode(
                self.netD1,
                perturbed_feature,
                output_size,
            )
            self.pred2 = self.decode(
                self.netD2,
                perturbed_feature,
                output_size,
            )
        else:
            self.pred1 = self.decode(
                self.netD1,
                feature,
                output_size,
            )
            self.pred2 = self.decode(
                self.netD2,
                feature,
                output_size,
            )

        self.pred = (self.pred1 + self.pred2) / 2.0
        self.mask = self.get_hard_label(self.pred)

    def calculate_losses(self):
        # 论文公式(3)：两个解码器分别计算监督分割损失。
        self.loss_seg1 = self.criterion_seg(
            self.pred1,
            self.gt,
        )
        self.loss_seg2 = self.criterion_seg(
            self.pred2,
            self.gt,
        )

        # 论文公式(1)：原始和扰动图像的特征一致性。
        self.loss_con = F.l1_loss(
            self.perturbed_feature,
            self.feature,
        ) * self.opt.lambda_con

        # 论文公式(2)：双解码器差异约束。
        self.loss_dis = (
            self.decoder_dissimilarity()
            * self.opt.lambda_dis
        )

        # 论文公式(4)：总目标。
        self.loss_total = (
            self.loss_seg1
            + self.loss_seg2
            + self.loss_con
            + self.loss_dis
        )

    def optimize(self):
        self.optimizer_MUTA.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available(),
            dtype=torch.float16,
        ):
            self.forward()
            self.calculate_losses()

        self.scaler.scale(self.loss_total).backward()
        self.scaler.step(self.optimizer_MUTA)
        self.scaler.update()

    def backward(self):
        # 训练入口由optimize统一处理AMP。
        self.calculate_losses()
        self.loss_total.backward()

    def visualization_preprocess(self):
        with torch.no_grad():
            self.color_gt = self.expand(self.gt)
            self.color_pred = self.expand(self.mask)
