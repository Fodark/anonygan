CUDA_VISIBLE_DEVICES=0
python train.py --batch_size 8 --arcface_arch r50 --lambda_gan 10 --lambda_pose 10 --lambda_wfm 5 --lambda_id 10 --lambda_rec 10 --same_percentage 1.0 --no_wfm