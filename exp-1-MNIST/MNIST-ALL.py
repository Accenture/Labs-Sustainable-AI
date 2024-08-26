import torch
# import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm # for progress bar
from argparse import ArgumentParser
import os
import time

# - Specific to MNIST - #
from fct_for_mnist import Net, train_0, create_dataloaders
# --------------------- #

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
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
                        # help='For Saving the current Model')

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


    args_parser = parser.parse_args()

    # --- FOR CALCULATORS 
    exp = ExpParams(args_parser)
    # -------------------

    #######################
    ##### Preparation #####
    #######################

    train = train_0 # don't display training stats
    print('---------------')
    print(exp.device_name == 'cuda')
    print('--------------')
    train_loader, test_loader = create_dataloaders(exp.device_name == 'cuda', args_parser)
    model = Net().to(exp.device)
    optimizer = optim.Adadelta(model.parameters(), lr=args_parser.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args_parser.gamma)


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

            train(args_parser, model, exp.device, train_loader, optimizer, epoch)
            scheduler.step()

            # --- FOR CALCULATORS 
            if exp.name_calc == "carbon_tracker":
                tracker.epoch_end()
            # -------------------

        model_path = os.path.join(args_parser.output_dir, "mnist_cnn.pt")
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
        PATH = os.path.join(_path, "models", "mnist_cnn.pt")
        model.load_state_dict(torch.load(PATH))    

        for kk in  tqdm(range(args_parser.nb_batch_inferences)):
            inputs, targets = next(iter(test_loader))
            inputs = inputs.to(exp.device)
            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)  
            # get the index of the max log-probability

        print("# -------------------------- #")
        print("# --- tag inference stop --- #")
        print("# -------------------------- #")
    

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
