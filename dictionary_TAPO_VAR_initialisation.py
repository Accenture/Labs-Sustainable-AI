import json

def dictionary_TAPO_VAR_initialisation():
    d = {
        "TAPORUN" : False, 
        "TAPOSAVED" : False
    }

    with open('TAPO-VAR.json', 'w') as f:
        json.dump(d, f, indent = 4, sort_keys=True)

    # with open('TAPO-VAR.json', 'r') as f:
    #     d = json.load(f)
    # print(d["TAPORUN"])

if __name__ == "__main__":
    dictionary_TAPO_VAR_initialisation()