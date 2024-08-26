import copy
import json
import os

def dictionary_list_initialisation(path):

    file1 = os.path.join(path, 'res_calc-time.json')
    file2 = os.path.join(path, 'res_meas-time.json')

    for_calculator = {
        "epochs" : [], 
        "time" : [], 
        "energy_consumed" : [], 
        "co2_emissions" : []}

    for_device = {
        "NOCALC": copy.deepcopy(for_calculator), 
        "CT:meas": copy.deepcopy(for_calculator), 
        "CT:pred": copy.deepcopy(for_calculator), 
        "GA:def": copy.deepcopy(for_calculator), 
        "GA:auto-para": copy.deepcopy(for_calculator), 
        "ECO2AI": copy.deepcopy(for_calculator), 
        "CC:on": copy.deepcopy(for_calculator), 
        "TAPO": copy.deepcopy(for_calculator),
        "FLOPS": copy.deepcopy(for_calculator)}

    for_experiment = {
        "linux_alienware": {
            "cuda": copy.deepcopy(for_device), 
            "cpu": copy.deepcopy(for_device)
        }
    }

    measurements = { 
        "mnist": {
            "training": copy.deepcopy(for_experiment), 
            "inference": copy.deepcopy(for_experiment)
        },
        "cifar10": {
            "training": copy.deepcopy(for_experiment), 
            "inference": copy.deepcopy(for_experiment)
        },
        "CUB_200_2011": {
            "training": copy.deepcopy(for_experiment), 
            "inference": copy.deepcopy(for_experiment)
        }, 
        "image_net": {
            "training": copy.deepcopy(for_experiment), 
            "inference": copy.deepcopy(for_experiment)
        }, 
        "SQUAD-extracted": {
            "training": copy.deepcopy(for_experiment), 
            "inference": copy.deepcopy(for_experiment)
        },
        "SQUAD-v1-1": {
            "training": copy.deepcopy(for_experiment), 
            "inference": copy.deepcopy(for_experiment)
        },
        "idle": {
            "training": copy.deepcopy(for_experiment), 
            "inference": copy.deepcopy(for_experiment)
        }
    }

    with open(file1, 'w') as f:
        json.dump(measurements, f, indent = 4, sort_keys=True)

    with open(file2, 'w') as f:
        json.dump(measurements, f, indent = 4, sort_keys=True)


if __name__ == "__main__":
    dictionary_list_initialisation('.')