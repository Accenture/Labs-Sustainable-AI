import torch
# import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm # for progress bar
from argparse import ArgumentParser
import os
import time

# --------------------- #

# --- FOR CALCULATORS
import sys
_path = '.'
sys.path.append(os.path.join(_path))
from fct_for_saving import save_cc
from fct_for_saving import save_ct
from fct_for_saving import save_eco2ai
from fct_for_saving import save_ga
from fct_for_saving import save_nocalc
from fct_for_saving import save_tapo
from fct_for_tapo import stop_TAPO
import psutil
import GPUtil
from fct_for_ga import stop_UTIL, mean_parallel_UTIL
from fct_for_experiments import ExpParams
from fct_for_experiments import prepare_calculator
from fct_for_experiments import start_calculators
from fct_for_experiments import stop_calculators
# ---------------------

def main():

    print("# ------------------------------ #")
    print("#                                #")
    print("#         -------------          #")
    print("#         --  START  --          #")
    print("#         -------------          #")
    print("#                                #")
    print("# ------------------------------ #")

    parser = ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 14)')
    # parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')

    parser.add_argument("--data_folder", type=str, help = "TODO", default = './data')
    parser.add_argument("--use_accelerator", type=str, help = "TODO", default = True)
    parser.add_argument("--save_model", type=str, help = "TODO", default = False)
    parser.add_argument("--ml_phase", type=str, help = "TODO", default = "inference")
    parser.add_argument("--calculator", type=str, help = "TODO", default = "no_calculator")
    parser.add_argument("--calculator_mode", type=str, help = "TODO", default = "")
    parser.add_argument("--dev_test", type=str, help = "TODO", default = "False")
    parser.add_argument("--nb_batch_inferences", type=int, help = "TODO", default = 100)
    parser.add_argument("--name_exp", type=str, help = "TODO", default = 'mnist')
    parser.add_argument("--computer", type=str, help = "TODO", default = 'linux_alienware')
    parser.add_argument("--path_logs_and_results", type=str, help = "TODO", default = '.')
    parser.add_argument("--output_dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--idle_time", default=60, type=int, help="duration of idle state tracking")

    # Calculators and modes:
    # code_carbon 
    # -> online,  offline
    # carbon_tracker 
    # -> measure, predict
    # eco2ai
    # energy_scopium
    # flops
    # green_algorithms
    # -> declarative, automated, automated_parallel
    # no_calculator
    # tapo

    args_parser = parser.parse_args()

    # --- FOR CALCULATORS 
    exp = ExpParams(args_parser)
    # -------------------
    
    #######################
    ##### Preparation #####
    #######################

    # --- FOR CALCULATORS 
    tracker = prepare_calculator(exp) 
    if exp.name_calc == 'green_algorithms':
        cpu_util = []
        gpu_util = []
        ram_util = []
    start_calculators(exp, tracker)
    t0 = time.time()
    # -------------------

    #####################
    #####   idle    #####
    #####################

    print("# ---------------------- #")
    print("# ---   idle start   --- #")
    print("# ---------------------- #")

    # --- FOR CALCULATORS 
    if exp.name_calc == "carbon_tracker":
        tracker.epoch_start()
    # -------------------   

    time.sleep(args_parser.idle_time)

    # --- FOR CALCULATORS 
    if exp.name_calc == "carbon_tracker":
        tracker.epoch_end()
    # -------------------

    print("# --------------------- #")
    print("# ---   idle stop   --- #")
    print("# --------------------- #")
    

    # --- FOR CALCULATORS
    tfinal = time.time()
    duration = tfinal - t0
    stop_calculators(exp, tracker)

    # Saving the data:
    if exp.name_calc == 'code_carbon':
        save_cc(exp, args_parser, duration)
    elif exp.name_calc == 'carbon_tracker':
        save_ct(exp, args_parser, duration)
    elif exp.name_calc == 'eco2ai':
        save_eco2ai(exp, args_parser, duration)
    elif exp.name_calc == 'green_algorithms':
        if exp.automated and exp.parallel:
            stop_UTIL(exp, t0, tfinal)
            cpu_util, gpu_util, ram_util = mean_parallel_UTIL(exp)
        save_ga(exp, args_parser, duration, 
            exp.automated, cpu_util, gpu_util, ram_util)
    elif exp.name_calc == 'tapo':
        stop_TAPO(exp, t0, tfinal)
        save_tapo(exp, args_parser)
    else: # no calculator
        save_nocalc(exp, args_parser, duration)
    # ----------------------


if __name__ == '__main__':
    main()
