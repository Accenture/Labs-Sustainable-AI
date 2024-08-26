import json

def dictionary_UTIL_VAR_initialisation():
    d = {
        "UTIL_RUN" : False, 
        "UTIL_SAVED" : False
    }

    with open('UTIL-VAR.json', 'w') as f:
        json.dump(d, f, indent = 4, sort_keys=True)

    # with open('TAPO-VAR.json', 'r') as f:
    #     d = json.load(f)
    # print(d["TAPORUN"])

if __name__ == "__main__":
    dictionary_UTIL_VAR_initialisation()