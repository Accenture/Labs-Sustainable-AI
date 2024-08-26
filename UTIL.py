import json
from datetime import datetime, timedelta
import os
import time
import psutil
import GPUtil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_logs_and_results", default="./other/tmp", type=str, help="dataset path")
args_parser = parser.parse_args()


# Initialise the dictionnary ..
with open('UTIL-VAR.json', 'r') as f:
    d = json.load(f)
d['UTIL_RUN']=False
d['UTIL_SAVED']=False
with open('UTIL-VAR.json', 'w') as f:
    json.dump(d, f)

print('Dictionnary initialized')

data = {}
time_list = []
cpu_util = []
gpu_util = []
ram_util = []

# .. and say that we are ready to record
d['UTIL_RUN']=True
with open('UTIL-VAR.json', 'w') as f:
    d = json.dump(d,f)
UTIL_RUN = True

print("UTIL READY")
print('UTIL RUN: ', UTIL_RUN)

while UTIL_RUN==True:
    time_s = time.time()
    time_list.append(time_s)
    cpu_util.append(psutil.cpu_percent())
    gpu_util.append(GPUtil.getGPUs()[0].load)
    ram_util.append(psutil.virtual_memory()[3]/10**9)
    time.sleep(2)
    
    with open('UTIL-VAR.json', 'r') as f:
        d = json.load(f)
    UTIL_RUN = d['UTIL_RUN']

data['cpu_util'] = cpu_util
data['gpu_util'] = gpu_util
data['ram_util'] = ram_util
data['time'] = time_list

# str_current_datetime = str(datetime.now())
str_current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
file_name = os.path.join(args_parser.path_logs_and_results,'util_logs', str_current_datetime+"_tool.txt")
with open(file_name, 'w') as f:
    json.dump(data, f, indent = 4, sort_keys=True)

# Indicate that we have finished saving the data
with open('UTIL-VAR.json', 'r') as f:
    d = json.load(f)
d['UTIL_SAVED']=True
with open('UTIL-VAR.json', 'w') as f:
    json.dump(d, f)

print("END-UTIL")