import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPSpawnPlugin

from src.models.anonygan import AnonyGAN
from src.data.dataloader import AnonyDataset
from src.options.train_options import TrainOptions


# model

if __name__ == "__main__":
    opt = TrainOptions().parse()

    model = AnonyGAN(opt)
    dataset = AnonyDataset(
        opt.data_root,
        opt.batch_size,
        opt.same_percentage,
        opt.train_same_identity,
        opt.reduced_landmarks,
    )

    logger = TensorBoardLogger(save_dir="./lightning_logs", version=opt.name)
    wandb_logger = WandbLogger()  # newline 2

    wandb_logger.watch(model.G, log="gradients", log_freq=100)
    wandb_logger.watch(model.D_gan, log="gradients", log_freq=100)
    wandb_logger.watch(model.D_pose, log="gradients", log_freq=100)

    # training
    trainer = pl.Trainer(
        gpus=-1,
        num_nodes=1,
        precision=32,
        limit_train_batches=1.0,
        max_epochs=700,
        logger=wandb_logger,
        accelerator="ddp_spawn",
        plugins=DDPSpawnPlugin(find_unused_parameters=True),
    )
    trainer.fit(model, dataset)
