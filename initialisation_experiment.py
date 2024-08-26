import os
from datetime import datetime
from dictionary_list_initialisation import dictionary_list_initialisation
from dictionary_TAPO_VAR_initialisation import dictionary_TAPO_VAR_initialisation
from dictionary_UTIL_VAR_initialisation import dictionary_UTIL_VAR_initialisation
import json


# Parent Directory path
parent_dir = "/home/demouser/Documents/Demos/energycalculatorsevaluation/logs_and_results"

# Directory
str_current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
directory = str_current_datetime + "_logs_and_results"

# Path
path = os.path.join(parent_dir, directory)

# Create the directory
os.mkdir(path)
print("Directory '% s' created" % directory)



# save directory name txt file
file_name = os.path.join(parent_dir, 'current_logs_and_results_folder.txt')
with open(file_name, 'w') as f:
    f.write(path)

# file_name = os.path.join(parent_dir, 'current_logs_and_results_folder.txt')
# with open(file_name, 'r') as f:
#     print(f.read())

# os.mkdir(os.path.join(path, 'carbon_tracker_logs'))
# os.mkdir(os.path.join(path, 'tapo_logs'))
# os.mkdir(os.path.join(path, 'term_logs'))
# os.mkdir(os.path.join(path, 'util_logs'))

dictionary_list_initialisation(path)
dictionary_TAPO_VAR_initialisation()
dictionary_UTIL_VAR_initialisation()


file = os.path.join(path, 'timestamps.json')
with open(file, 'w') as f:
    json.dump({}, f, indent = 4)
