from PyP100 import PyP110
import base64
import json
import datetime
from pprint import pprint
import time 
# from multiprocessing import Process, Event
from datetime import datetime, timedelta
import os
import numpy as np


def find_parallel_records_data(folder_logs):
    """ find the data corresponing to the latest experiment """
    list_files_tool = []
    list_files_task = []
    for file in os.listdir(folder_logs):
        if file.endswith("tool.txt"):
            list_files_tool.append(file)
        if file.endswith("task.txt"):
            list_files_task.append(file)

    tmp_file_tool = max(list_files_tool)
    print(tmp_file_tool)

    tmp_file_task = max(list_files_task)
    print(tmp_file_task)

    data_tool_file = os.path.join(folder_logs, tmp_file_tool)
    data_task_file = os.path.join(folder_logs, tmp_file_task)

    with open(data_tool_file, 'r') as f:
        data_tool = json.load(f)

    with open(data_task_file, 'r') as f:
        data_task = json.load(f)

    return(data_tool, data_task)


def select_data_UTIL(t_start, t_end, data_util):

    cpu_util_list = np.array(data_util['cpu_util']) 
    gpu_util_list = np.array(data_util['gpu_util'])
    ram_util_list = np.array(data_util['ram_util'])
    time_list = np.array(data_util['time'])
    
    idx1 = time_list > t_start 
    idx2 = time_list < t_end
    idx = np.array([idx1[k] and idx2[k] for k in range(len(idx1))])
    
    time_list_task = time_list[idx]
    cpu_util_list_task = cpu_util_list[idx]
    gpu_util_list_task = gpu_util_list[idx]
    ram_util_list_task = ram_util_list[idx]

    cpu_util_mean = np.mean(cpu_util_list_task)
    gpu_util_mean = np.mean(gpu_util_list_task)
    ram_util_mean = np.mean(ram_util_list_task)
    
    return(cpu_util_mean, gpu_util_mean, ram_util_mean)


def wait_for_UTIL():
    with open('UTIL-VAR.json', 'r') as f:
        d = json.load(f)
    UTIL_RUN = d['UTIL_RUN']
    while UTIL_RUN == False:
        with open('UTIL-VAR.json', 'r') as f:
            d = json.load(f)
        UTIL_RUN = d['UTIL_RUN']
        time.sleep(1)


def stop_UTIL(exp, t0, tfinal):
    with open('UTIL-VAR.json', 'r') as f:
        d = json.load(f)
    d['UTIL_RUN']=False
    with open('UTIL-VAR.json', 'w') as f:
        json.dump(d, f)    

    data = {}
    data[exp.ml + '_start'] = t0
    data[exp.ml + '_end'] = tfinal
    str_current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = os.path.join(exp.path_logs_and_results,'util_logs', str_current_datetime+"_task.txt")
    with open(file_name, 'w') as f:
        json.dump(data, f, indent = 4, sort_keys=True)    


def mean_parallel_UTIL(exp):

    folder_traces = os.path.join(exp.path_logs_and_results, 'util_logs')

    UTIL_SAVED = False
    while UTIL_SAVED == False:
        with open('UTIL-VAR.json', 'r') as f:
            d = json.load(f)
        UTIL_SAVED = d['UTIL_SAVED']
        time.sleep(1)
    print(UTIL_SAVED)

    data_util, data_task = find_parallel_records_data(folder_traces)
    print('')
    print('Data UTIL: ', data_util)
    print('')
    print('Data ML Task: ', data_task)

    t0 = data_task[exp.ml+'_start']
    tfinal = data_task[exp.ml+'_end']
    
    cpu_util, gpu_util, ram_util = select_data_UTIL(t0, tfinal, data_util)
    return(cpu_util, gpu_util, ram_util)