import json 
import os
import pandas as pd
from datetime import timedelta
import numpy as np
from fct_for_tapo import find_tapo_data, select_data
import time
from carbontracker import parser as CTparser
from fct_for_ES import fct_time_energy, fct_find_data
from ga_automatic_request import do_request

def _save(file_name, exp, meas_epochs, time, meas_energy, meas_co2):

    name_exp = exp.name
    ml = exp.ml
    comp = exp.comp
    dev = exp.device_name


    calc_short = {'code_carbon:online':'CC:on', 
                'code_carbon:offline':'CC:off',
                'carbon_tracker:measure':'CT:meas',
                'carbon_tracker:predict':'CT:pred',
                'eco2ai':'ECO2AI', 
                'energy_scopium':'ES', 
                'flops':'FLOPS', 
                'green_algorithms:default':'GA:def', 
                'green_algorithms:automated':'GA:auto',
                'green_algorithms:automated_parallel':'GA:auto-para', 
                'tapo':'TAPO',
                'no_calculator':'NOCALC'}
    
    key = exp.name_calc
    if key == 'code_carbon':
        if exp.online == True:
            key = key + ':' + 'online'
        else:
            key = key + ':' + 'offline'
    elif key == 'carbon_tracker':
        if exp.measure == True:
            key = key + ':' + 'measure'
        else:
            key = key + ':' + 'predict'
    elif key == 'green_algorithms':
        if exp.automated == False:
            key = key + ':' + 'default'
        else:
            if exp.parallel == True:
                key = key + ':' + 'automated_parallel'
            else:
                key = key + ':' + 'automated'

    print('---------------------> key', key)
    calc = calc_short[key]
    
    print('# -------------- #')
    print('# --- Saving --- #')
    print('# -------------- #')
    
    print('Experience is: ', name_exp)
    print('ML phase is: ', ml)
    print('Computer is: ',comp)
    print('Torch device is: ', dev)
    print('Calculator is: ', calc)

    with open(file_name, 'r') as f:
        measurements_dict = json.load(f)

        my_info = measurements_dict[name_exp][ml][comp][dev][calc]
        print('Data before: ', my_info)

        my_info["epochs"].append(meas_epochs)
        my_info["time"].append(time)
        my_info["energy_consumed"].append(meas_energy)
        my_info["co2_emissions"].append(meas_co2)

        print('Data after: ', my_info)
        
    with open(file_name, 'w') as f:
        json.dump(measurements_dict, f, indent = 4, sort_keys=True)

def save_data(exp, meas_epochs, meas_time, calc_time, meas_energy, meas_co2):

    tmp = exp.path_logs_and_results
    tmp = tmp.split("/")
    print(tmp)
    tmp = tmp[:-1]
    tmp = "/".join(tmp)
    print(tmp)

    file1 = os.path.join(tmp, 'res_calc-time.json')
    file2 = os.path.join(tmp, 'res_meas-time.json')

    print('# --- Saving time by hand --- #')
    _save(file2, exp, meas_epochs, meas_time, meas_energy, meas_co2)

    print('# --- Saving time by calc --- #')
    if calc_time != None:
        _save(file1, exp, meas_epochs, calc_time, meas_energy, meas_co2)


# ------------------- #
# --- Code Carbon --- #
# ------------------- #

def save_cc(exp, args_parser, duration):
    # Saving the data in the json file

    output_file_name = exp.cc_output_file
    file = pd.read_csv(output_file_name)
    df=pd.DataFrame(file)

    meas_epochs = exp.epochs

    if exp.ml == 'training': 
        meas_time = duration
        calc_time = df["duration"].iloc[-1]
        meas_energy = df["energy_consumed"].iloc[-1]
        meas_co2 = df["emissions"].iloc[-1]
    elif exp.ml == 'inference':
        nb_inferences = args_parser.nb_batch_inferences
        meas_time = duration/nb_inferences
        calc_time = df["duration"].iloc[-1]/nb_inferences
        meas_energy = df["energy_consumed"].iloc[-1]/nb_inferences
        meas_co2 = df["emissions"].iloc[-1]/nb_inferences
    save_data(exp, meas_epochs, meas_time, calc_time, meas_energy, meas_co2)
    
    # os.remove(output_file_name)


# ---------------------- #
# --- Carbon Tracker --- #
# ---------------------- #

def save_ct(exp, args_parser, t):
    # Saving the data in the json file

    log_dir = exp.ct_log_dir
    meas_epochs = exp.epochs

    PUE_2022 = 1.59

    nb_inferences = args_parser.nb_batch_inferences
    if exp.ml == "inference":
        factor = nb_inferences
    else:
        factor = 1

    # Results from the parser
    logs = CTparser.parse_all_logs(log_dir=log_dir)
    first_log = logs[0]
    print(first_log)
    d1 = first_log['actual']
    if exp.measure == True:
        print ("{:<22} {:<15}".format('','Measured'))
    else:
        d1 = first_log['pred']
        print ("{:<22} {:<15}".format('','Predicted'))

    meas_time = t/factor
    calc_time = d1["duration (s)"]/factor
    meas_energy = d1["energy (kWh)"]/factor/PUE_2022
    meas_co2 = d1["co2eq (g)"]/factor*10**(-3)/PUE_2022
    if meas_co2 != meas_co2:
        meas_co2 = "N/A"
    

    save_data(exp, meas_epochs, meas_time, calc_time, meas_energy, meas_co2)


    for k in d1.keys():
        if k == "equivalents":
            if d1[k] != None: 
                for kk in d1[k].keys():
                    s0 = kk
                    s1 = d1[k][kk]
                    s1 = float(s1)/factor
                    print("{:<22} {:<15}".format(s0, s1))
        elif k == 'epochs' and exp.ml == "inference":
            s0 = k
            s1 = "N/A"
            print("{:<22} {:<15}".format(s0, s1))
        else:
            s0 = k
            s1 = d1[k]
            s1 = float(s1)/factor
            print("{:<22} {:<15}".format(s0, s1))

# ------------------- #
# ---   Eco2AI    --- #
# ------------------- #

def save_eco2ai(exp, args_parser, duration):
    # Saving the data in the json file 
    
    output_file = exp.eco2ai_output_file
    file = pd.read_csv(output_file)
    df=pd.DataFrame(file)
    meas_epochs = exp.epochs

    if exp.ml == 'training':
        meas_time = duration
        calc_time = df["duration(s)"].iloc[-1]
        meas_energy = df["power_consumption(kWh)"].iloc[-1]
        meas_co2 = df["CO2_emissions(kg)"].iloc[-1]
    elif exp.ml == 'inference':
        nb_inferences = args_parser.nb_batch_inferences
        meas_time = duration/nb_inferences
        calc_time = df["duration(s)"].iloc[-1]/nb_inferences
        meas_energy = df["power_consumption(kWh)"].iloc[-1]/nb_inferences
        meas_co2 = df["CO2_emissions(kg)"].iloc[-1]/nb_inferences

    save_data(exp, meas_epochs, meas_time, calc_time, meas_energy, meas_co2)

    # os.remove(output_file)


# --------------------- #
# --- EnergyScopium --- #
# --------------------- #

def save_ES(exp, args_parser, duration):
    # Saving the data in the json file

    # ES data corresponding to this experiment
    # folder_traces = os.environ['ENERGY_SCOPE_TRACES_PATH']
    folder_traces = os.environ['ENERGYSCOPIUM_TRACES_PATH']
    ES_data = fct_find_data(folder_traces)
    meas_co2 = "N/A"
    meas_epochs = exp.epochs

    if exp.ml == "training":
        tag = 'tag_training'
        meas_time = duration
        calc_time, meas_energy = fct_time_energy(ES_data, tag)
    elif exp.ml == 'inference':
        nb_inferences = args_parser.nb_batch_inferences
        tag = 'tag_inference'
        meas_time = duration/nb_inferences
        calc_time, meas_energy = fct_time_energy(ES_data, tag)
        calc_time = meas_time/nb_inferences
        meas_energy = meas_energy/nb_inferences

    save_data(exp, meas_epochs, meas_time, calc_time, meas_energy, meas_co2)


# ------------------------ #
# --- Green Algorithms --- #
# ------------------------ #

def save_ga(exp, args_parser, duration, 
            util_tracking, cpu_util, gpu_util, ram_util):
    # Saving the data in the json file

    calc_time = None
    meas_epochs = exp.epochs

    if exp.ml == 'training':
        meas_time = duration
        factor = 1
    elif exp.ml == 'inference':
        nb_inferences = args_parser.nb_batch_inferences
        factor = nb_inferences

    td = timedelta(seconds = duration)
    td = str(td)
    print('td         : ', td)
    td = td.split(':')
    print('split td   : ', td)
    td_h = td[0]      # hours as int
    print('hours      : ', td_h)
    td_m = td[1]      # minutes as int
    print('minutes    : ', td_m)
    td_sm = td[2]     # seconds as floating numbers
    print('seconds    : ', td_sm)
    td_s = str(float(td_sm)/60)   # seconds as fractions of minutes 
    print('frac of min: ', td_s)
    td_m = float(td_m)+ float(td_s)
    print('float min  : ', td_m)
    runTime_hour_input = int(td_h)
    runTime_min_input = td_m

    print('Hours      :', runTime_hour_input)
    print('Minutes    :', runTime_min_input)

    GA_inputs = {}

    if util_tracking:   
        usageCPU_radio = "Yes"
        usageCPU_input = np.mean(cpu_util)/100 # CPU utl mean
        usageGPU_radio = "Yes"
        usageGPU_input = np.mean(gpu_util) # GPU utl mean
        usageRAM_input = np.mean(ram_util) # RAM utl mean

        print('cpu usage  : ', usageCPU_input)
        print('gpu usage  : ', usageGPU_input)
        print('ram usage  : ', usageRAM_input)

        GA_inputs['runTime_hour_input']=runTime_hour_input
        GA_inputs['runTime_min_input']=runTime_min_input
        GA_inputs['usageCPU_radio']=usageCPU_radio
        GA_inputs['usageCPU_input']=usageCPU_input
        GA_inputs['usageGPU_radio']=usageGPU_radio
        GA_inputs['usageGPU_input']=usageGPU_input
        GA_inputs['usageRAM_input']=usageRAM_input

        try:
            output_GA = do_request(runTime_hour_input = runTime_hour_input, 
                                    runTime_min_input = runTime_min_input,
                                    usageCPU_radio = usageCPU_radio,
                                    usageCPU_input = usageCPU_input,
                                    usageGPU_radio = usageGPU_radio,
                                    usageGPU_input = usageGPU_input,
                                    memory_input = usageRAM_input)
        except Exception as e:
            print('Communication with GA failed: ', e)
            file_name_ga = os.path.join(exp.path_logs_and_results, 'GA_inputs.json')
            with open(file_name_ga, 'w') as f:
                json.dump(GA_inputs, f, indent = 4)
    else:
        GA_inputs['runTime_hour_input']=runTime_hour_input
        GA_inputs['runTime_min_input']=runTime_min_input

        try:
            output_GA = do_request(runTime_hour_input = runTime_hour_input, 
                                    runTime_min_input = runTime_min_input)
        except Exception as e:
            print('Communication with GA failed: ', e)
            file_name_ga = os.path.join(exp.path_logs_and_results, 'GA_inputs.json')
            with open(file_name_ga, 'w') as f:
                json.dump(GA_inputs, f, indent = 4)
        
    meas_time = duration/factor
    meas_energy = output_GA["energy_needed"]/factor
    meas_co2 = output_GA["carbonEmissions"]*10**(-3)/factor
    save_data(exp, meas_epochs, meas_time, calc_time, meas_energy, meas_co2)
    

# ------------------- #
# ---    TAPO     --- #
# ------------------- #

def save_tapo(exp, args_parser):
    # Saving the data in the json file

    folder_traces = os.path.join(exp.path_logs_and_results,'tapo_logs')

    with open('TAPO-VAR.json', 'r') as f:
        d = json.load(f)
    TAPOSAVED = d['TAPOSAVED']
    print(TAPOSAVED)
    while TAPOSAVED == False:
        with open('TAPO-VAR.json', 'r') as f:
            d = json.load(f)
        TAPOSAVED = d['TAPOSAVED']
        time.sleep(1)
    print(TAPOSAVED)

    data_tapo, data_task = find_tapo_data(folder_traces)
    print('')
    print('Data TAPO: ', data_tapo)
    print('')
    print('Data ML Task: ', data_task)

    t0 = data_task[exp.ml+'_start']
    tf = data_task[exp.ml+'_end']
    duration = tf - t0
    Ec_kWh = select_data(t0, tf, data_tapo)
    print('Energy consumed: ', Ec_kWh)
    calc_time = None
    meas_co2 = "N/A"
    meas_epochs = exp.epochs

    if exp.ml == "training":
        meas_time = duration
        meas_energy = Ec_kWh
    elif exp.ml == "inference":
        # nb_inferences = args.nb_batch_inferences*test_batch_size
        nb_inferences = args_parser.nb_batch_inferences
        meas_time = duration/nb_inferences
        meas_energy = Ec_kWh/nb_inferences

    save_data(exp, meas_epochs, meas_time, calc_time, meas_energy, meas_co2)


# ------------------- #
# ---    FLOPs    --- #
# ------------------- #

def save_FLOPS(exp, args_parser, Ec_kWh):
    # Saving the data in the json file
    
    meas_time = "N/A"
    calc_time = None
    meas_co2 = "N/A"
    meas_energy = Ec_kWh
    meas_epochs = exp.epochs

    save_data(exp, meas_epochs, meas_time, calc_time, meas_energy, meas_co2)
    

# --------------------- #
# --- No calculator --- #
# --------------------- #

def save_nocalc(exp, args_parser, duration):
    # Saving the data in the json file

    meas_energy = "N/A"
    meas_co2 = "N/A"
    calc_time = None
    meas_epochs = exp.epochs

    if exp.ml == "training":
        meas_time = duration
    elif exp.ml == "inference":
        # nb_inferences = args.nb_batch_inferences*test_batch_size
        nb_inferences = args_parser.nb_batch_inferences
        meas_time = duration/nb_inferences

    save_data(exp, meas_epochs, meas_time, calc_time, meas_energy, meas_co2)