from PyP100 import PyP110
import base64
import json
import datetime
from pprint import pprint
from time import time, sleep
# from multiprocessing import Process, Event
from datetime import datetime, timedelta
import os
import numpy as np

def find_tapo_data(folder_tapo_logs):
    """ find the data corresponing to the latest experiment """
    list_files_tapo = []
    list_files_task = []
    for file in os.listdir(folder_tapo_logs):
        if file.endswith("tapo.txt"):
            list_files_tapo.append(file)
        if file.endswith("task.txt"):
            list_files_task.append(file)

    tmp_file_tapo = max(list_files_tapo)
    print(tmp_file_tapo)

    tmp_file_task = max(list_files_task)
    print(tmp_file_task)

    data_tapo_file = os.path.join(folder_tapo_logs, tmp_file_tapo)
    data_task_file = os.path.join(folder_tapo_logs, tmp_file_task)

    with open(data_tapo_file, 'r') as f:
        data_tapo = json.load(f)

    with open(data_task_file, 'r') as f:
        data_task = json.load(f)

    return(data_tapo, data_task)


def select_data(t_start, t_end, data_tapo):
    time_list = np.array(data_tapo['time'])
    power_list = np.array(data_tapo['power'])
    idx1 = time_list > t_start 
    idx2 = time_list < t_end
    idx = np.array([idx1[k] and idx2[k] for k in range(len(idx1))])
    time_list_training = time_list[idx]
    power_list_training = power_list[idx]
    Ec_J = np.trapz(power_list_training, x=time_list_training)
    Ec_kWh = Ec_J/(3.6*10**6)
    return(Ec_kWh)

def wait_for_TAPO():
    with open('TAPO-VAR.json', 'r') as f:
        d = json.load(f)
    TAPORUN = d['TAPORUN']
    while TAPORUN == False:
        with open('TAPO-VAR.json', 'r') as f:
            d = json.load(f)
        TAPORUN = d['TAPORUN']
        sleep(1)

def stop_TAPO(exp, t0, tfinal):
    with open('TAPO-VAR.json', 'r') as f:
        d = json.load(f)
    d['TAPORUN']=False
    with open('TAPO-VAR.json', 'w') as f:
        json.dump(d, f)    

    data = {}
    data[exp.ml + '_start'] = t0
    data[exp.ml + '_end'] = tfinal
    str_current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = os.path.join(exp.path_logs_and_results,'tapo_logs', str_current_datetime+"_task.txt")
    with open(file_name, 'w') as f:
        json.dump(data, f, indent = 4, sort_keys=True)