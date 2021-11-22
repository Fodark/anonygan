import os
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from PIL import Image

from .utils import SpecificNorm, get_current_visuals, tensor2im
from . import ArcFace, Generator, Discriminator


class AnonyGAN(pl.LightningModule):
    def __init__(self, opt=None):
        super().__init__()
        self.automatic_optimization = False

        self.G = Generator(
            ch_input=opt.ch_input,
            use_ch_att=not opt.no_ch_att,
            reduced_landmarks=opt.reduced_landmarks,
        )
        # self.D_rf = Discriminator(3)
        self.D_gan = Discriminator(3 + 3)
        if opt.double_discriminator:
            self.D_gan2 = Discriminator(3)
        if opt.reduced_landmarks:
            self.D_pose = Discriminator(3 + 29)
        else:
            self.D_pose = Discriminator(3 + 68)
        if opt.pretrained_id_discriminator:
            self.D_id = ArcFace(opt.arcface_arch)
        elif opt.id_discriminator:
            self.D_id = Discriminator(3 + 3)

        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.pose_criterion = nn.BCEWithLogitsLoss()
        if opt.pretrained_id_discriminator:
            self.id_criterion = nn.CosineSimilarity()
        elif opt.id_discriminator:
            self.id_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()
        self.wfm_criterion = nn.L1Loss()
        self.sp = SpecificNorm()
        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

        self.opt = opt

        self.real_label = torch.ones((opt.batch_size, 1, 11, 9))
        self.fake_label = torch.zeros((opt.batch_size, 1, 11, 9))

        self.images_dir = os.path.join("./images", opt.name)
        os.makedirs(self.images_dir, exist_ok=True)

    def forward(self, x):
        return None

    def lambda_rule(self, epoch):
        lr_l = 1.0 - max(0, epoch + 1 + 1 - 500) / float(200 + 1)
        return lr_l

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.G.parameters(), lr=self.opt.g_lr, betas=(self.opt.beta1, 0.999)
        )
        opt_d_gan = torch.optim.Adam(
            self.D_gan.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )

        sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, self.lambda_rule)
        sch_d_gan = torch.optim.lr_scheduler.LambdaLR(opt_d_gan, self.lambda_rule)

        if self.opt.double_discriminator:
            opt_d_gan2 = torch.optim.Adam(
                self.D_gan2.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            sch_d_gan2 = torch.optim.lr_scheduler.LambdaLR(opt_d_gan2, self.lambda_rule)
        opt_d_pose = torch.optim.Adam(
            self.D_pose.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
        )
        sch_d_pose = torch.optim.lr_scheduler.LambdaLR(opt_d_pose, self.lambda_rule)

        if self.opt.id_discriminator:
            opt_d_id = torch.optim.Adam(
                self.D_id.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
            )
            sch_d_id = torch.optim.lr_scheduler.LambdaLR(opt_d_id, self.lambda_rule)

        opts, schs = [opt_g, opt_d_gan, opt_d_pose], [sch_g, sch_d_gan, sch_d_pose]

        if self.opt.id_discriminator:
            opts.append(opt_d_id)
            schs.append(sch_d_id)

        if self.opt.double_discriminator:
            opts.append(opt_d_gan2)
            schs.append(sch_d_gan2)

        return opts, schs


    def training_step(self, train_batch, batch_idx):
        self.real_label = self.real_label.to(self.device)
        self.fake_label = self.fake_label.to(self.device)

        opts = self.optimizers()
        opt_g, opt_d_gan1, opt_d_pose = opts[:3]
        if self.opt.double_discriminator:
            opt_d_gan2 = opts[-1]
        if self.opt.id_discriminator:
            opt_d_id = opts[3]

        input_P1, input_BP1 = train_batch["P1"], train_batch["BP1"]
        masked_P2, input_BP2 = train_batch["masked_P2"], train_batch["BP2"]
        input_P2 = train_batch["P2"]
        same = train_batch["same"]

        if self.opt.ch_input == 6:
            G_input = [
                torch.cat((input_P1, masked_P2), dim=1),
                torch.cat((input_BP1, input_BP2), dim=1),
            ]
        else:
            G_input = [
                input_P1,
                torch.cat((input_BP1, input_BP2), dim=1),
            ]

        fake_P2 = self.G(G_input)
        if self.opt.ch_input == 3:
            fake_P2 = (1.0 - train_batch["mask"]) * input_P2 + train_batch[
                "mask"
            ] * fake_P2

        # === TRAINING G ===
        opt_g.zero_grad()

        ## real/fake
        fake_pair = torch.cat((fake_P2, input_P1), dim=1)
        real_pair = torch.cat((input_P2, input_P1), dim=1)
        gan_real1 = self.D_gan(real_pair)

        if self.opt.double_discriminator:
            gan_real2 = self.D_gan2(input_P1)  # input_P2
            real_features = [gan_real2]
        else:
            real_features = [gan_real1]

        gan_fake1 = self.D_gan(fake_pair)

        if self.opt.double_discriminator:
            gan_fake2 = self.D_gan2(fake_P2)
            fake_features = [gan_fake2]
        else:
            fake_features = [gan_fake1]

        app_loss = (
            self.gan_criterion(gan_fake1[-1], self.real_label) * self.opt.lambda_gan
        )

        gan_loss = 0.0
        if self.opt.double_discriminator:
            gan_loss = (
                self.gan_criterion(gan_fake2[-1], self.real_label) * self.opt.lambda_gan
            )
            self.log("G/RF", gan_loss.item(), prog_bar=True)
        # else:
        #     gan_loss2 = app_loss
        # gan_loss = (app_loss + gan_loss2) / 2
        self.log("G/APP", app_loss.item(), prog_bar=True)

        ## WFM
        wfm_loss = 0.0
        if not self.opt.no_wfm:
            n_layers_D = 4

            if self.opt.yiming:
                gan_real2 = gan_real1
                gan_fake2 = gan_fake1

            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0
            for j in range(0, len(gan_fake2) - 1):
                tmp_loss = (
                    D_weights
                    * feat_weights
                    * self.wfm_criterion(gan_fake2[j], gan_real2[j].detach())
                    * self.opt.lambda_wfm
                )
                wfm_loss += tmp_loss

            self.log("G/WFM", wfm_loss.item(), prog_bar=True)

        ## pose
        pose_fake = self.D_pose(torch.cat((fake_P2, input_BP2), 1))[-1]
        pose_loss = (
            self.pose_criterion(pose_fake, self.real_label) * self.opt.lambda_pose
        )
        self.log("G/POSE", pose_loss.item())

        ## id
        id_loss = 0.0
        if self.opt.pretrained_id_discriminator:
            condition_resized = F.interpolate(input_P1, (112, 112))
            condition_resized = self.sp(condition_resized)
            fake_resized = F.interpolate(fake_P2, (112, 112))
            fake_resized = self.sp(fake_resized)
            id_embedding_real = self.D_id(condition_resized)
            id_embedding_fake = self.D_id(fake_resized)

            id_loss = (
                1 - self.id_criterion(id_embedding_real, id_embedding_fake).mean()
            ) * self.opt.lambda_id
            self.log("G/ID", id_loss.item())

        if self.opt.id_discriminator:
            fake_pair = torch.cat([input_P1, fake_P2], dim=1)
            id_fake = self.D_id(fake_pair)[-1].mean(dim=[1, 2, 3])
            id_loss = self.id_criterion(id_fake, self.real_label) * self.opt.lambda_id
            self.log("G/ID", id_loss.item(), prog_bar=True)

        ## reconstruction
        rec_loss = 0.0
        if not self.opt.no_l1 and self.opt.train_same_identity:
            generated_P2 = (1.0 - train_batch["mask"]) * input_P2 + train_batch[
                "mask"
            ] * fake_P2
            fake_filtered, real_filtered = generated_P2[same], input_P2[same]
            rec_loss = (
                self.l1_criterion(fake_filtered, real_filtered) * self.opt.lambda_rec
            )
            self.log("G/L1", rec_loss.item(), prog_bar=True)

        total_loss = gan_loss + app_loss + wfm_loss + pose_loss + id_loss + rec_loss

        self.manual_backward(total_loss)
        opt_g.step()

        # === TRAINING D RF

        if self.opt.double_discriminator:
            opt_d_gan2.zero_grad()

            fea2_real = self.D_gan2(input_P1)[-1]  # input_P2
            fea2_fake = self.D_gan2(fake_P2.detach())[-1]

            loss_D_real2 = self.gan_criterion(fea2_real, self.real_label)
            loss_D_fake2 = self.gan_criterion(fea2_fake, self.fake_label)

            D_loss_real = (loss_D_real2 + loss_D_fake2) / 2
            self.log("D/RF", D_loss_real.item())

            self.manual_backward(D_loss_real)
            opt_d_gan2.step()

        # === TRAINING D APP ===

        opt_d_gan1.zero_grad()

        # D_Fake
        fake_pair = torch.cat((fake_P2.detach(), input_P1), dim=1)
        fea1_fake = self.D_gan(fake_pair)[-1]
        loss_D_fake1 = self.gan_criterion(fea1_fake, self.fake_label)

        # D_Real
        real_pair = torch.cat((input_P2, input_P1), dim=1)
        fea1_real = self.D_gan(real_pair)[-1]
        loss_D_real1 = self.gan_criterion(fea1_real, self.real_label)

        D_app_loss = (loss_D_fake1 + loss_D_real1) / 2 * self.opt.lambda_gan
        self.log("D/APP", D_app_loss.item())

        self.manual_backward(D_app_loss)
        opt_d_gan1.step()

        # === TRAINING D POSE ===
        opt_d_pose.zero_grad()

        fea1_fake = self.D_pose(torch.cat((fake_P2.detach(), input_BP2), 1))[-1]
        loss_D_fake1 = self.gan_criterion(fea1_fake, self.fake_label)

        # D_Real
        fea1_real = self.D_pose(torch.cat((input_P2, input_BP2), 1))[-1]
        loss_D_real1 = self.gan_criterion(fea1_real, self.real_label)

        D_pose_loss = (loss_D_fake1 + loss_D_real1) / 2 * self.opt.lambda_pose
        self.log("D/POSE", D_pose_loss.item())

        self.manual_backward(D_pose_loss)
        opt_d_pose.step()

        # === TRAINING D ID IF NEEDED ===
        if self.opt.id_discriminator:
            opt_d_id.zero_grad()

            fake_pair = torch.cat([input_P1, fake_P2.detach()], dim=1)
            real_pair = torch.cat([input_P1, input_P2], dim=1)

            fea1_fake = self.D_id(fake_pair)[-1].mean(dim=[1, 2, 3])
            loss_D_fake1 = self.gan_criterion(fea1_fake, self.fake_label)

            # D_Real
            fea1_real = self.D_id(real_pair)[-1].mean(dim=[1, 2, 3])
            loss_D_real1 = self.gan_criterion(fea1_real, self.real_label)

            D_id_loss = (loss_D_fake1 + loss_D_real1) / 2 * self.opt.lambda_id
            self.log("D/ID", D_id_loss.item())

            self.manual_backward(D_id_loss)
            opt_d_id.step()

        # end of batch, schedule lr
        for sch in self.lr_schedulers():
            sch.step(self.current_epoch)

    def validation_step(self, train_batch, batch_idx):
        input_P1, input_BP1 = train_batch["P1"], train_batch["BP1"]
        masked_P2, input_BP2 = train_batch["masked_P2"], train_batch["BP2"]
        input_P2 = train_batch["P2"]

        if self.opt.ch_input == 6:
            G_input = [
                torch.cat((input_P1, masked_P2), dim=1),
                torch.cat((input_BP1, input_BP2), dim=1),
            ]
        else:
            G_input = [
                input_P1,
                torch.cat((input_BP1, input_BP2), dim=1),
            ]

        fake_P2 = self.G(G_input)
        generated_P2 = (1.0 - train_batch["mask"]) * input_P2 + train_batch[
            "mask"
        ] * fake_P2  # masked_P2 + train_batch["mask"] * fake_P2
        viz = get_current_visuals(
            input_P1, input_P2, masked_P2, input_BP1, input_BP2, fake_P2, generated_P2
        )
        img = Image.fromarray(viz)
        img.save(os.path.join(self.images_dir, str(self.current_epoch) + ".jpg"))

    def test_step(self, test_batch, batch_idx):
        input_P1, input_BP1 = test_batch["P1"], test_batch["BP1"]
        masked_P2, input_BP2 = test_batch["masked_P2"], test_batch["BP2"]
        input_P2 = test_batch["P2"]
        P1_name, P2_name = test_batch["P1_path"], test_batch["P2_path"]

        if self.opt.ch_input == 6:
            G_input = [
                torch.cat((input_P1, masked_P2), dim=1),
                torch.cat((input_BP1, input_BP2), dim=1),
            ]
        else:
            G_input = [
                input_P1,
                torch.cat((input_BP1, input_BP2), dim=1),
            ]

        fake_P2 = self.G(G_input)
        generated_P2 = (1.0 - test_batch["mask"]) * input_P2 + test_batch[
            "mask"
        ] * fake_P2
        # print('f', fake_P2.shape)
        img = Image.fromarray(tensor2im(fake_P2.data))
        img.save(
            os.path.join(
                self.opt.output_path,
                P2_name[0] + ".jpg"
                # P1_name[0] + "_" + P2_name[0].split("/")[1] + ".jpg",
            )
        )
        #
        # viz = get_current_visuals(
        #    input_P1, input_P2, masked_P2, input_BP1, input_BP2, fake_P2, generated_P2
        # )
        # img = Image.fromarray(viz)
        # img.save(os.path.join(self.images_dir, str(self.current_epoch) + ".jpg"))