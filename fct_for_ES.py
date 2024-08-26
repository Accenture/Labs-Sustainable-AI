import json 
import os
import pprint
import datetime

def fct_find_data(folder_traces):
    """ find the data corresponing to the latest experiment """
    list_dir = []
    for file in os.listdir(folder_traces):
        d = os.path.join(folder_traces, file)
        if os.path.isdir(d):
            if file.startswith("tmp"):
                list_dir.append(file)

    tmp_file = max(list_dir)

    data_file = folder_traces+"/"+tmp_file+"/"+"energy_scope_eprofile_0.txt"

    with open(data_file, 'r') as f:
        ES_data = json.load(f)

    return(ES_data)



def fct_time_energy(ES_data, tag):
    """ returns the duration in seconds of the tagged task """

    ES_tag_data = ES_data['data']['data']['tags'][tag]


    date_str2tuple = lambda st : tuple([int(x) for x in st[:10].split('/')])+tuple([int(x) for x in st[11:19].split(':')])
    stop_tuple= date_str2tuple(ES_tag_data['stop'])
    # stop_sec_float = float('0.'+ES_tag_data['stop'][20:])

    start_tuple= date_str2tuple(ES_tag_data['start'])
    # start_sec_float = float('0.'+ES_tag_data['start'][20:])

    stop_datetime = datetime.datetime(*stop_tuple)
    start_datetime = datetime.datetime(*start_tuple)

    meas_time = (stop_datetime - start_datetime).total_seconds()

    meas_energy = float(ES_tag_data['joule(J)'])/3.6*10**(-6)  # kWh

    return(meas_time, meas_energy)
    



# # print(datetime.strftime(ES_time_data['stop'] ))

# # meas_time = ES_time_data['stop'] - ES_time_data['start']

# # meas_epochs = epochs
# # meas_time = 
# # meas_energy = 
# # meas_co2 = 

# # '2023/01/03 17:42:06.414168'

# with open(folder_traces+'/energy_scope_enode_0_demouser-Alienware-Aurora-R9.txt', 'r') as f:
#     text = json.load(f)
# pprint.pprint(text)
