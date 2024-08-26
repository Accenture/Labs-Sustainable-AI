import pickle
import os
import pandas as pd
import json

"""
Store all Scholar results in a single csv files.
Removes duplicates within Scholar resutls.
"""

if __name__ == "__main__":
    f_name = os.path.join("sentences_scholar","sentence-scholar.json")
    with open(f_name, 'r') as f:
        S_dict = json.load(f)


    fold = os.path.join("saved_results_scholar","csv_all")
    f_0 = "scholar_results_"

    nb_q = 103

    total_nb_res = 0
    for i in range(1, nb_q+1):
        f_name = f_0+f"{i:03}.csv"
        if f_name in os.listdir(fold):

            print(f"x --- Query {i:03} --- x")
            print("query: ", S_dict[str(i)])
            print("file name: ", f_name)
            f_path = os.path.join(fold, f_name)
            pd_file = pd.read_csv(f_path)
            current_df=pd.DataFrame(pd_file)
            print("number results: ", current_df.shape[0])
            total_nb_res += current_df.shape[0]

            if 'concat_df' not in vars():
                concat_df = current_df
            else:
                concat_df = pd.concat([concat_df, current_df], ignore_index=True)
        

    print("Total number of results: ", total_nb_res)

    # Check the duplicates:
    # dupl = concat_df[concat_df.duplicated(subset = 'result_id', keep=False)][['result_id', 'title']]
    # dupl.sort_values("result_id", inplace=True)
    # dupl.to_csv('test.csv')

    # Remove the duplicates:
    no_dupl_df = concat_df.drop_duplicates(subset='result_id', keep='first')
    print(no_dupl_df.shape[0])
    f_path = os.path.join("saved_results_scholar", "scholar_results.csv")
    no_dupl_df.to_csv(f_path)
    
    # concat_df.sort_values("result_id", inplace=True)


    # cpt = 0
    # folder = "pickle_save"
    # for file in os.listdir(folder):
    #     cpt += 1
    # for idx in range(1, cpt):
    #     print("x --------------- x")
    #     file_name = os.path.join("pickle_save", f"scholar_results_{idx}.pkl")
    #     print(file_name)
    #     f = open(file_name, 'rb')
    #     df = pickle.load(f)
    #     f.close()
    
    
    # file_name = os.path.join("tests", "test_serpapi.pkl")
    # f = open(file_name, 'rb')
    # df = pickle.load(f)
    # f.close()
    # print(df.columns)
        