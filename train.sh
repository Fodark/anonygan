# python train.py --batch_size 32 --ch_input 3 --lambda_gan 5 --lambda_pose 5 --lambda_wfm 5 --lambda_id 10 --lambda_rec 10 --same_percentage 1.0 --no_wfm --no_l1 --name exp0base --data_root /datasets/anonygan-dataset

# This is secret and shouldn't be checked into version control
export WANDB_API_KEY=""
# Name and notes optional
export WANDB_NAME="AnonyGAN"
# export WANDB_NOTES="Smaller learning rate for D no ID disc more percentage lower lambda gan wfm and no chatt and no mask"
# Only needed if you don't checkin the wandb/settings file
export WANDB_ENTITY="f0dark"
export WANDB_PROJECT="anonygan"
export WANDB_START_METHOD="fork"
# python train.py --batch_size 32 --ch_input 3 --lr 2e-6 --lambda_gan 1 --lambda_pose 1 --lambda_wfm 1 --lambda_id 5 --lambda_rec 10 --train_same_identity --same_percentage 0.75 --yiming --no_ch_att --no_wfm --name hao --data_root /datasets/anonygan-dataset


python train.py --batch_size 32 --ch_input 6 --lr 2e-6 --lambda_gan 1 --lambda_pose 1 --lambda_wfm 1 --lambda_id 5 --lambda_rec 10 --train_same_identity --same_percentage 0.75 --yiming --no_ch_att --no_wfm --reduced_landmarks --name hao_mask_29kp --data_root /datasets/anonygan-dataset