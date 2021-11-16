import pytorch_lightning as pl
import os

from train import AnonyGAN
from src.data.dataloader import AnonyDataset
from src.options.test_options import TestOptions


def test(opt):
    model = AnonyGAN.load_from_checkpoint(opt.ckpt, opt=opt, strict=False)
    # print("RD", opt.reduced_landmarks)
    dataset = AnonyDataset(
        opt.data_root,
        opt.batch_size,
        opt.same_percentage,
        opt.train_same_identity,
        opt.reduced_landmarks,
        opt.iciap,
        opt.lfw,
    )

    # training
    trainer = pl.Trainer(
        gpus=-1,
        num_nodes=1,
        precision=32,
    )
    trainer.test(model=model, datamodule=dataset)


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.double_discriminator = False
    opt.pretrained_id_discriminator = False
    opt.id_discriminator = False
    opt.same_percentage = 0.0
    opt.train_same_identity = True

    opt.output_path = os.path.join("output", "ablation", opt.name)
    os.makedirs(opt.output_path, exist_ok=True)

    test(opt)
