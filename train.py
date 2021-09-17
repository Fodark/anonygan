import os
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image

from src.models import Generator, Discriminator, ArcFace
from src.models.utils import _gradient_penalty, SpecificNorm, get_current_visuals
from src.data.dataloader import AnonyDataset
from src.options.train_options import TrainOptions


class AnonyGAN(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.automatic_optimization = False

        self.G = Generator()
        self.D_gan = Discriminator(3 + 3)
        if opt.double_discriminator:
            self.D_gan2 = Discriminator(3)
        self.D_pose = Discriminator(3 + 68)
        if opt.pretrained_id_discriminator:
            self.D_id = ArcFace(opt.arcface_arch)

        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.pose_criterion = nn.BCEWithLogitsLoss()
        if opt.pretrained_id_discriminator:
            self.id_criterion = nn.CosineSimilarity()
        self.l1_criterion = nn.L1Loss()
        self.wfm_criterion = nn.L1Loss()
        self.sp = SpecificNorm()
        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

        self.opt = opt

        self.real_label = torch.ones(opt.batch_size)
        self.fake_label = torch.zeros(opt.batch_size)

        self.images_dir = os.path.join("./images", opt.name)
        os.makedirs(self.images_dir, exist_ok=True)

    def forward(self, x):
        return None

    def lambda_rule(self, epoch):
        lr_l = 1.0 - max(0, epoch + 1 + 1 - 100) / float(100 + 1)
        return lr_l

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.G.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999)
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

        if self.opt.double_discriminator:
            return [opt_g, opt_d_gan, opt_d_gan2, opt_d_pose], [
                sch_g,
                sch_d_gan,
                sch_d_gan2,
                sch_d_pose,
            ]
        else:
            return [opt_g, opt_d_gan, opt_d_pose], [sch_g, sch_d_gan, sch_d_pose]

    def training_step(self, train_batch, batch_idx):
        self.real_label = self.real_label.to(self.device)
        self.fake_label = self.fake_label.to(self.device)
        if self.opt.double_discriminator:
            opt_g, opt_d_gan1, opt_d_gan2, opt_d_pose = self.optimizers()
        else:
            opt_g, opt_d_gan1, opt_d_pose = self.optimizers()

        input_P1, input_BP1 = train_batch["P1"], train_batch["BP1"]
        masked_P2, input_BP2 = train_batch["masked_P2"], train_batch["BP2"]
        input_P2 = train_batch["P2"]
        # same_identity = train_batch["same"]

        G_input = [
            torch.cat((input_P1, masked_P2), dim=1),
            torch.cat((input_BP1, input_BP2), dim=1),
        ]

        fake_P2 = self.G(G_input)
        # same_identity_indices = same_identity == True

        if self.opt.double_discriminator:
            img_fake_downsample = self.downsample(fake_P2)
            img_real_downsample = self.downsample(input_P2)

        # === TRAINING G ===
        opt_g.zero_grad()

        ## real/fake
        fake_pair = torch.cat((fake_P2, input_P1), dim=1)
        real_pair = torch.cat((input_P2, input_P1), dim=1)
        gan_real1 = self.D_gan(real_pair)

        if self.opt.double_discriminator:
            gan_real2 = self.D_gan2(img_real_downsample)
            real_features = [gan_real1, gan_real2]
        else:
            real_features = [gan_real1]

        gan_fake1 = self.D_gan(fake_pair)
        # print("GAN FAKE 1", gan_fake1[-1].mean(dim=[1, 2, 3]))
        if self.opt.double_discriminator:
            gan_fake2 = self.D_gan2(img_fake_downsample)
            fake_features = [gan_fake1, gan_fake2]
        else:
            fake_features = [gan_fake1]

        gan_loss1 = (
            self.gan_criterion(gan_fake1[-1].mean(dim=[1, 2, 3]), self.real_label)
            * self.opt.lambda_gan
        )
        # print("GAN LOSS 1", gan_loss1)
        if self.opt.double_discriminator:
            gan_loss2 = (
                self.gan_criterion(gan_fake2[-1].mean(dim=[1, 2, 3]), self.real_label)
                * self.opt.lambda_gan
            )
        else:
            gan_loss2 = gan_loss1
        gan_loss = (gan_loss1 + gan_loss2) / 2
        self.log("G/APP", gan_loss.item(), prog_bar=True)

        ## WFM
        wfm_loss = 0.0
        if not self.opt.no_wfm:
            n_layers_D = 4
            num_D = 2 if self.opt.double_discriminator else 1
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            for i in range(num_D):
                for j in range(0, len(fake_features[i]) - 1):
                    wfm_loss += (
                        D_weights
                        * feat_weights
                        * self.wfm_criterion(
                            fake_features[i][j], real_features[i][j].detach()
                        )
                        * self.opt.lambda_wfm
                    )
            self.log("G/WFM", wfm_loss.item())

        ## pose
        pose_fake = self.D_pose(torch.cat((fake_P2, input_BP2), 1))[-1].mean(
            dim=[1, 2, 3]
        )
        pose_loss = (
            self.pose_criterion(pose_fake, self.real_label) * self.opt.lambda_pose
        )
        self.log("G/PB", pose_loss.item())

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

        ## reconstruction
        rec_loss = 0.0
        if not self.opt.no_l1:
            rec_loss = self.l1_criterion(fake_P2, input_P2) * self.opt.lambda_rec
            self.log("G/L1", rec_loss.item())

        total_loss = gan_loss + wfm_loss + pose_loss + id_loss + rec_loss

        self.manual_backward(total_loss)
        opt_g.step()

        # === TRAINING D GAN ===
        opt_d_gan1.zero_grad()
        if self.opt.double_discriminator:
            opt_d_gan2.zero_grad()

        fake_pair = torch.cat((fake_P2.detach(), input_P1), dim=1)
        fea1_fake = self.D_gan(fake_pair)[-1].mean(dim=[1, 2, 3])
        if self.opt.double_discriminator:
            fea2_fake = self.D_gan2(img_fake_downsample.detach())[-1].mean(
                dim=[1, 2, 3]
            )

        loss_D_fake1 = self.gan_criterion(fea1_fake, self.fake_label)
        if self.opt.double_discriminator:
            loss_D_fake2 = self.gan_criterion(fea2_fake, self.fake_label)
            D_gan_loss_fake = (loss_D_fake1 + loss_D_fake2) / 2
        else:
            D_gan_loss_fake = loss_D_fake1

        # D_Real
        real_pair = torch.cat((input_P2, input_P1), dim=1)
        fea1_real = self.D_gan(real_pair)[-1].mean(dim=[1, 2, 3])
        if self.opt.double_discriminator:
            fea2_real = self.D_gan2(img_real_downsample)[-1].mean(dim=[1, 2, 3])

        loss_D_real1 = self.gan_criterion(fea1_real, self.real_label)
        if self.opt.double_discriminator:
            loss_D_real2 = self.gan_criterion(fea2_real, self.real_label)
            D_gan_loss_real = (loss_D_real1 + loss_D_real2) / 2
        else:
            D_gan_loss_real = loss_D_real1

        D_gan_loss = (D_gan_loss_fake + D_gan_loss_real) / 2 * self.opt.lambda_gan
        self.log("D/APP", D_gan_loss.item())

        self.manual_backward(D_gan_loss)
        opt_d_gan1.step()
        if self.opt.double_discriminator:
            opt_d_gan2.step()

        # === TRAINING D POSE ===
        opt_d_pose.zero_grad

        fea1_fake = self.D_pose(torch.cat((fake_P2.detach(), input_BP2), 1))[-1].mean(
            dim=[1, 2, 3]
        )
        loss_D_fake1 = self.gan_criterion(fea1_fake, self.fake_label)

        # D_Real
        fea1_real = self.D_pose(torch.cat((input_P2, input_BP2), 1))[-1].mean(
            dim=[1, 2, 3]
        )
        loss_D_real1 = self.gan_criterion(fea1_real, self.real_label)

        D_pose_loss = (loss_D_fake1 + loss_D_real1) / 2 * self.opt.lambda_pose

        self.log("D/POSE", D_pose_loss.item())

        self.manual_backward(D_pose_loss)
        opt_d_pose.step()

    def on_train_epoch_end(self):
        for sch in self.lr_schedulers():
            sch.step()

    def validation_step(self, train_batch, batch_idx):
        input_P1, input_BP1 = train_batch["P1"], train_batch["BP1"]
        masked_P2, input_BP2 = train_batch["masked_P2"], train_batch["BP2"]
        input_P2 = train_batch["P2"]

        G_input = [
            torch.cat((input_P1, masked_P2), dim=1),
            torch.cat((input_BP1, input_BP2), dim=1),
        ]

        fake_P2 = self.G(G_input)
        generated_P2 = masked_P2 + train_batch["mask"] * fake_P2
        viz = get_current_visuals(
            input_P1, input_P2, masked_P2, input_BP1, input_BP2, fake_P2, generated_P2
        )
        img = Image.fromarray(viz)
        img.save(os.path.join(self.images_dir, str(self.current_epoch) + ".jpg"))


# model

if __name__ == "__main__":
    opt = TrainOptions().parse()

    model = AnonyGAN(opt)
    dataset = AnonyDataset(opt.data_root, opt.batch_size, opt.same_percentage)

    logger = TensorBoardLogger(save_dir="./lightning_logs", version=opt.name)

    # training
    trainer = pl.Trainer(
        gpus=-1,
        num_nodes=1,
        precision=32,
        limit_train_batches=1.0,
        max_epochs=700,
        logger=logger,
    )
    trainer.fit(model, dataset)
