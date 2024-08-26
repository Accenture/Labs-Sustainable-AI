import torch
# import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser
import os
import time

# - Specific to CIFAR10 - #
from fct_for_cifar10 import Net, train_0, create_dataloaders
from torch.nn import CrossEntropyLoss
# ----------------------- #

# --- FOR CALCULATORS
import sys
_path = '.'
sys.path.append(os.path.join(_path))
from tqdm import tqdm
from fct_for_saving import save_cc
from fct_for_saving import save_ct
from fct_for_saving import save_eco2ai
from fct_for_saving import save_FLOPS
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
from fct_for_experiments import flops_method_pytorch
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

    parser.add_argument("--batch_size", type=int, help = "TODO", default = 4)
    parser.add_argument("--test_batch_size", type=int, help = "TODO", default = 4)
    parser.add_argument("--lr", type=float, help = "TODO", default = 0.001)
    parser.add_argument("--momentum", type=float, help = "TODO", default = 0.9)
    parser.add_argument("--seed", type=float, help = "TODO", default = 1)
    parser.add_argument("--data_folder", type=str, help = "TODO", default = './data')

    parser.add_argument("--use_accelerator", type=str, help = "TODO", default = True)
    parser.add_argument("--save_model", type=str, help = "TODO", default = False)
    parser.add_argument("--ml_phase", type=str, help = "TODO", default = "inference")
    parser.add_argument("--calculator", type=str, help = "TODO", default = "no_calculator")
    parser.add_argument("--calculator_mode", type=str, help = "TODO", default = "")
    parser.add_argument("--dev_test", type=str, help = "TODO", default = "False")
    parser.add_argument("--nb_batch_inferences", type=int, help = "TODO", default = 1000)
    parser.add_argument("--name_exp", type=str, help = "TODO", default = 'cifar10')
    parser.add_argument("--computer", type=str, help = "TODO", default = 'linux_alienware')
    parser.add_argument("--epochs", type=int, help = "TODO", default = 10)
    parser.add_argument("--path_logs_and_results", type=str, help = "TODO", default = '.')
    parser.add_argument("--output_dir", default=".", type=str, help="path to save outputs")

    # ---------------------

    args_parser = parser.parse_args()
  
    # --- FOR CALCULATORS 
    exp = ExpParams(args_parser)
    # -------------------


    #####################
    #### Preparation ####
    #####################

    train = train_0 # don't display training stats
    train_loader, test_loader = create_dataloaders(args_parser)
    model = Net().to(exp.device)
    criterion = CrossEntropyLoss() # Define a Loss function and optimizer below
    optimizer = optim.SGD(model.parameters(), lr=args_parser.lr, momentum=args_parser.momentum)
    # Classification Cross-Entropy loss and SGD with momentum


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
    ##### Training  #####
    #####################

    if (exp.ml == "training") and exp.name_calc != 'flops':

        print("# ---------------------- #")
        print("# --- training start --- #")
        print("# ---------------------- #")

        for epoch in tqdm(range(1, exp.epochs + 1)):
            
            # --- FOR CALCULATORS 
            if exp.name_calc == "carbon_tracker":
                tracker.epoch_start()
            # -------------------             

            train(model, exp.device, train_loader, criterion, optimizer, epoch)

            # --- FOR CALCULATORS 
            if exp.name_calc == "carbon_tracker":
                tracker.epoch_end()
            # -------------------

        # if exp.save_model:
        model_path = os.path.join(args_parser.output_dir, "cifar_net.pth")
        torch.save(model.state_dict(), model_path)
        print("# --------------------- #")
        print("# --- training stop --- #")
        print("# --------------------- #")


    #####################
    ##### Inference #####
    #####################

    if (exp.ml == "inference") and not exp.name_calc == 'flops':

        print("# ----------------------- #")
        print("# --- inference start --- #")
        print("# ----------------------- #")

        # recover the saved model:
        PATH = os.path.join(_path, "models", "cifar_net.pth")
        model.load_state_dict(torch.load(PATH))    

        for kk in  tqdm(range(args_parser.nb_batch_inferences)):
            input, targets = next(iter(test_loader))
            input = input.to(exp.device)
            output = model(input)
            _, pred = torch.max(output, 1)

            # --- FOR CALCULATORS 
            if exp.name_calc == 'green_algorithms' and exp.automated and (not exp.parallel):
                cpu_util.append(psutil.cpu_percent())
                gpu_util.append(GPUtil.getGPUs()[0].load)
                ram_util.append(psutil.virtual_memory()[3]/1000000000)
            # ------------------

        print("# ---------------------- #")
        print("# --- inference stop --- #")
        print("# ---------------------- #")
    

    # --- FOR CALCULATORS
    tfinal = time.time()
    duration = tfinal - t0
    stop_calculators(exp, tracker)
    copy_model = model.to(torch.device("cpu"))
    Ec_kWh = flops_method_pytorch(exp, train_loader, copy_model)

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
    elif exp.name_calc =='flops':
        save_FLOPS(exp, args_parser, Ec_kWh)
    else: # no calculator
        save_nocalc(exp, args_parser, duration)
    # ----------------------

if __name__ == '__main__':
    main()