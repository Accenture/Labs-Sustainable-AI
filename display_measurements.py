import copy
import pprint
import json

with open('measurements_data.json', 'r') as f:
    measurements = json.load(f)
    pprint.pprint(measurements)

# device = 'cuda'
# computer = 'linux_alienware'

# with open('measurements_data.json', 'w') as f:
#     my_info = measurements["mnist"]["training"][computer][device]["CT"]
#     print(my_info)

#     my_info["time"] = 999

#     print(my_info)

#     json.dump(measurements, f, indent = 4, sort_keys=True)

# with open('measurements_data.json', 'r') as f:
#     mmm = json.load(f)
#     print(mmm["mnist"]["training"][computer][device]["CT"])
#     pprint.pprint(mmm)