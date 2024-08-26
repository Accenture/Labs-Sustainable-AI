import requests
import json
import sys
import os
from pprint import pprint
_path = '.'
sys.path.append(os.path.join(_path))

def do_request(
    runTime_hour_input: int = 12,
    runTime_min_input: float = 0.0,
    coreType_dropdown: str = 'Both',
    CPUmodel_dropdown: str = 'other',
    numberCPUs_input: int = 8,
    tdpCPU_input: int = 95/8,
    GPUmodel_dropdown: str = 'other',
    numberGPUs_input: int = 1,
    tdpGPU_input: int = 250,
    memory_input: int = 64,
    location_country_dropdown: str = 'France',
    usageCPU_radio: str = "No",
    usageCPU_input: float = 1.0,
    usageGPU_radio: str = "No",
    usageGPU_input: float = 1.0,
    platform_type: str = "personalComputer",
) -> dict:

    # Load default json file
    f = open(os.path.join(_path, 'GA_request_data.json'))
    data = json.load(f)

    # Modify values if needed
    data['inputs'][1]['value'] = coreType_dropdown
    data['inputs'][2]['value'] = numberCPUs_input
    data['inputs'][3]['value'] = CPUmodel_dropdown
    data['inputs'][5]['value'] = tdpCPU_input
    data['inputs'][6]['value'] = numberGPUs_input
    data['inputs'][7]['value'] = GPUmodel_dropdown
    data['inputs'][9]['value'] = tdpGPU_input
    data['inputs'][10]['value'] = memory_input
    data['inputs'][11]['value'] = runTime_hour_input
    data['inputs'][12]['value'] = runTime_min_input
    data['inputs'][14]['value'] = location_country_dropdown
    data['inputs'][20]['value'] = usageCPU_radio
    data['inputs'][21]['value'] = usageCPU_input
    data['inputs'][22]['value'] = usageGPU_radio
    data['inputs'][23]['value'] = usageGPU_input
    data['inputs'][29]['value'] = platform_type

    # Perform request
    response = requests.post(
        'http://calculator.green-algorithms.org/_dash-update-component',
        json=data,
        verify=False,
    )

    # Prepare out in json format
    response_json = response.json()

    output_json = {
        "energy_needed": response_json["response"]["aggregate_data"]["data"]["energy_needed"],
        "carbonEmissions": response_json["response"]["aggregate_data"]["data"]["carbonEmissions"]
    }

    return output_json