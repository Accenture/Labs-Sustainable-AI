from PyP100 import PyP110
import base64
import json
import datetime
from time import time, sleep
from datetime import datetime, timedelta
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_logs_and_results", default="./other/tmp", type=str, help="dataset path")
args_parser = parser.parse_args()


with open('TAPO-VAR.json', 'r') as f:
    d = json.load(f)
d['TAPORUN']=False
d['TAPOSAVED']=False
with open('TAPO-VAR.json', 'w') as f:
    json.dump(d, f)

with open('TAPO-credentials.json', 'r') as f:
    cred = json.load(f)
plugip = cred["ip"]
username = cred["email"]
password = cred["password"]
p110 = PyP110.P110(plugip, username, password)

p110.handshake()
p110.login()

returnedData = p110.getDeviceInfo()
if 'result' not in returnedData.keys():
    returnedData = {'result' : returnedData}
nickname = base64.b64decode(returnedData['result']['nickname'])
nicknameDecoded = nickname.decode("utf-8")
print ("Plug Name:", nicknameDecoded)
print ("Device IP:", returnedData['result']['ip'])
print ("Device On:", returnedData['result']['device_on'])
print ("Device Model:", returnedData['result']['model'])
print ("Firmware Ver:", returnedData['result']['fw_ver'])
print ("Device ID:", returnedData['result']['device_id'])
print ("MAC:", returnedData['result']['mac'])
print ("Device On Time:", (timedelta(seconds=(returnedData['result']['on_time']))))
print ("Device Overheated:", returnedData['result']['overheated'])
print ("Power Protection:", returnedData['result']['power_protection_status'])
print ("RSSI:", returnedData['result']['rssi'])
print ("Signal Level:", returnedData['result']['signal_level'])
print (" ")

print("TAPO-READY")
d['TAPORUN']=True
with open('TAPO-VAR.json', 'w') as f:
    d = json.dump(d,f)
TAPORUN = True

data = {}
power_list = []
time_list = []
date_list = []

print('xxxxxxxxxxxxxxxxxxxxxxxx')
returnedData = {}
returnedData['result'] = p110.getEnergyUsage()
print(returnedData)
print('xxxxxxxxxxxxxxxxxxxxxxxx')

while TAPORUN==True:

    time_s = time()
    try:
        returnedData = p110.getEnergyUsage()
        if 'result' not in returnedData.keys():
            returnedData = {'result' : returnedData}
        print(returnedData)
        power_list.append(returnedData['result']['current_power'] / 1000)
        time_list.append(time_s)
        date_list.append(returnedData['result']['local_time'])
        sleep(2)
    except Exception as e:
        print("Couldn't retrieve info this time: ", time_s, " error: ", e)

    with open('TAPO-VAR.json', 'r') as f:
        d = json.load(f)
    TAPORUN = d['TAPORUN']

data['power'] = power_list
data['time'] = time_list
data['date'] = date_list

str_current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
file_name = os.path.join(args_parser.path_logs_and_results,'tapo_logs', str_current_datetime+"_tapo.txt")
with open(file_name, 'w') as f:
    json.dump(data, f, indent = 4, sort_keys=True)


with open('TAPO-VAR.json', 'r') as f:
    d = json.load(f)
d['TAPOSAVED']=True
with open('TAPO-VAR.json', 'w') as f:
    json.dump(d, f)

print("END-TAPO")