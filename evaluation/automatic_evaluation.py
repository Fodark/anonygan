import subprocess, shutil, os
import argparse

celeba_root = "/media/dvl1/SSD_DATA/BiGraphCeleba/"
lfw_root = "/media/dvl1/SSD_DATA/celeba-gcn-lfw/"

celeba_output = "/media/dvl1/SSD_DATA/BiGraphCeleba/misc_outputs/"
lfw_output = "/media/dvl1/SSD_DATA/celeba-gcn-lfw/"

def run_evaluation(opt):
    datasets = ["celeba", "lfw"]
    c_out = os.path.join(celeba_output, opt.exp_name)
    os.makedirs(c_out, exist_ok=True)

    celeba_command = f"python ../celeba/test.py --dataroot {celeba_root} --name {opt.exp_name} --model BiGraphGAN --phase test --dataset_mode keypoint --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc {opt.kp} --no_flip --which_model_netG Graph --checkpoints_dir ./checkpoints --pairLst {celeba_root}celeba-pairs-test.csv --which_epoch latest --results_dir ./results/ --display_id 0 --P_input_nc {6 if opt.mask else 3} --dataset celeba --out_dir {c_out}"
    celeba_command += " --use_fuser" if opt.fuser else ""

    res = subprocess.call(celeba_command, shell = True)
    if res != 0:
        print("Something went wrong with celeba...")
        exit(res)
    
    l_out = os.path.join(lfw_output, f"output-{opt.exp_name}")
    os.makedirs(l_out, exist_ok=True)

    lfw_command = f"python ../celaba/test.py --dataroot {lfw_root} --name {opt.exp_name} --model BiGraphGAN --phase test --dataset_mode keypoint --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc {opt.kp} --no_flip --which_model_netG Graph --checkpoints_dir ./checkpoints --pairLst {lfw_root}test_pairs.csv --which_epoch latest --results_dir ./results/ --display_id 0 --P_input_nc {6 if opt.mask else 3} --dataset lfw --out_dir {l_out}"
    lfw_command += " --use_fuser" if opt.fuser else ""

    res = subprocess.call(lfw_command, shell = True)
    if res != 0:
        print("Something went wrong with lfw...")
        exit(res)
    
    eval_command = f"python evaluate.py --img_dir {c_out} --dataset /media/dvl1/SSD_DATA/BiGraphCeleba/test --lfw_generated {l_out} --exp_name {opt.exp_name}"
    
    res = subprocess.call(eval_command, shell = True)
    if res != 0:
        print("Something went wrong with evaluation...")
        exit(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, required=True)
    # parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--kp", type=int, required=True)
    parser.add_argument("--opencv", action="store_true")
    parser.add_argument("--fuser", action="store_true")
    parser.add_argument("--mask", action="store_true")

    opt = parser.parse_args()

    run_evaluation(opt)