import os

from test import test
from evaluate import compute_metrics
from src.options.test_options import TestOptions

orig = "/media/dvl1/HDD_DATA/iciap/ciagan_lfw/lfw-only-second"

models = [
    # {"name": "ciagan-resized", "to_run": False, "orig": orig},
    # {"name": "simswap", "to_run": False, "orig": orig},
    # {
    #     "name": "hao_orig_29",
    #     "to_run": False,
    #     "input_ch": 3,
    #     "rd": True,
    #     "ckpt": "ckpts/hao_original_29kp.ckpt",
    #     "orig": orig,
    # },
    {
        "name": "hao_mask_29",
        "to_run": True,
        "input_ch": 6,
        "rd": True,
        "ckpt": "ckpts/hao_original_mask_29kp.ckpt",
        "orig": orig,
        "lfw": False,
    },
    {
        "name": "hao_mask_68",
        "to_run": True,
        "input_ch": 6,
        "rd": False,
        "ckpt": "ckpts/hao_original_mask_68kp.ckpt",
        "orig": orig,
        "lfw": False,
    },
    {
        "name": "hao_mask_chatt",
        "to_run": True,
        "input_ch": 6,
        "rd": False,
        "ckpt": "ckpts/hao_yiming_no_wfm.ckpt",
        "orig": orig,
        "lfw": False,
    },
    {
        "name": "hao_yiming",
        "to_run": True,
        "input_ch": 6,
        "rd": False,
        "ckpt": "/home/dvl1/projects/anonygan/ckpts/hao_yiming.ckpt",
        "orig": orig,
        "lfw": False,
    },
]

for m in models:
    output_path = os.path.join("/media/dvl1/HDD_DATA/iciap/outputs-fair", m["name"])
    if m["to_run"]:
        opt = TestOptions().parse()
        opt.double_discriminator = False
        opt.pretrained_id_discriminator = False
        opt.id_discriminator = False
        opt.same_percentage = 0.0
        opt.train_same_identity = True
        opt.ch_input = m["input_ch"]
        opt.reduced_landmarks = m["rd"]
        opt.ckpt = m["ckpt"]
        opt.iciap = True
        opt.output_path = output_path
        opt.name = m["name"]
        opt.lfw = m["lfw"]
        os.makedirs(opt.output_path, exist_ok=True)
        test(opt)
    # print(f'=== TESTING {m["name"]} ===')
    # compute_metrics(m["orig"], output_path)
    # print()
