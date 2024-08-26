from serpapi import GoogleSearch
from urllib.parse import urlsplit, parse_qsl
import pandas as pd
import pickle
import os
import json

def organic_results(query, api_key, start):
    print("extracting organic results..")

    params = {
        "api_key": api_key,          # https://serpapi.com/manage-api-key
        "engine": "google_scholar",
        "q": query,  # search query
        "hl": "en",        # language
        "start": start,      # first page
        "hl": "en",
        "num": "20",
        "as_vis": "1"
    }

    search = GoogleSearch(params)

    organic_results_data = []

    loop_is_true = True

    cpt = 1
    while loop_is_true:


      results = search.get_dict()
      if "error" in results:
        no_res_msg = "Google hasn't returned any results for this query."
        if results["error"] == no_res_msg:
          print("No results!")
          return "no_results"
      if "organic_results" not in results:
        print("x ------------- x")
        print("Tockens finished!")
        print("Position: ", position)
        print("Result ID: ", result_id)
        print("x ------------- x")
        return organic_results_data

      # Setting the number of the current page:
      if cpt == 1:
        try:
          page_nb = results["serpapi_pagination"]["current"]
          page_nb_temp = page_nb
        except Exception: # only 1 page
          page_nb = 1
      else:
        try:
          page_nb = results["serpapi_pagination"]["current"]
          page_nb_temp = page_nb
        except Exception: # current page is the last page
          page_nb = page_nb_temp + 1

      print(f"Currently extracting page №{page_nb}..")


      for result in results["organic_results"]:
          position = result["position"]
          title = result["title"]
          publication_info_summary = result["publication_info"]["summary"]
          result_id = result["result_id"]
          link = result.get("link")
          result_type = result.get("type")
          snippet = result.get("snippet")

          try:
            file_title = result["resources"][0]["title"]
          except: file_title = None

          try:
            file_link = result["resources"][0]["link"]
          except: file_link = None

          try:
            file_format = result["resources"][0]["file_format"]
          except: file_format = None

          try:
            cited_by_count = int(result["inline_links"]["cited_by"]["total"])
          except: cited_by_count = None

          cited_by_id = result.get("inline_links", {}).get("cited_by", {}).get("cites_id", {})
          cited_by_link = result.get("inline_links", {}).get("cited_by", {}).get("link", {})

          try:
            total_versions = int(result["inline_links"]["versions"]["total"])
          except: total_versions = None

          all_versions_link = result.get("inline_links", {}).get("versions", {}).get("link", {})
          all_versions_id = result.get("inline_links", {}).get("versions", {}).get("cluster_id", {})
          
          organic_results_data.append({
            "page_number": page_nb,
            "position": position + 1,
            "result_type": result_type,
            "title": title,
            "link": link,
            "result_id": result_id,
            "publication_info_summary": publication_info_summary,
            "snippet": snippet,
            "cited_by_count": cited_by_count,
            "cited_by_link": cited_by_link,
            "cited_by_id": cited_by_id,
            "total_versions": total_versions,
            "all_versions_link": all_versions_link,
            "all_versions_id": all_versions_id,
            "file_format": file_format,
            "file_title": file_title,
            "file_link": file_link,
          })

      try:
        if "next" in results.get("serpapi_pagination", {}):
            search.params_dict.update(dict(parse_qsl(urlsplit(results["serpapi_pagination"]["next"]).query)))
        else:
            loop_is_true = False
      except Exception: # current page is the last page
        loop_is_true = False

      cpt+=1

      # else: # there is no result at all
      #   print("No results!")
      #   organic_results_data = "no_results"
      #   loop_is_true = False

    return organic_results_data

if __name__ == '__main__':
  # data_list = organic_results()
  # print("waiting for organic results to save..")
  # data_df = pd.DataFrame(data=data_list)
  # data_df.to_csv("google_scholar_organic_results.csv", encoding="utf-8", index=False)
  # data_df.to_pickle("test_serpapi.pkl")

  # file_sentences = os.path.join("sentences_scholar", 'sentence-scholar.pkl')
  # file = open(file_sentences, 'rb')
  # S = pickle.load(file)
  # file.close()

  f_name = os.path.join("sentences_scholar","sentence-scholar.json")
  with open(f_name, 'r') as f:
    S_dict = json.load(f)


  key_1 = ""
  key_2 = ""
  key_3 = ""
  key_4 = ""
  key_5 = ""
  key_6 = ""
  folder_corresp = {key_1:"key_1", 
                    key_2:"key_2",
                    key_3:"key_3",
                    key_4:"key_4",
                    key_5:"key_5",
                    key_6:"key_6"}

  api_key = ""           # <-- enter your key
  query = 1              # <-- choose query
  start_sentence = query
  end_sentence = query
  start_page = "0"        # <-- start result number
  second_part = False     # <--
  if second_part == True:
    str_end = '_end'
  else:
    str_end = ''


  for cpt in range(start_sentence, end_sentence+1):
    print("x --------------- x")
    print(f"x --- String {cpt} --- x")
    print("x --------------- x")
    key = str(cpt)
    s = S_dict[key]
    print(key, s)
    
    folder = folder_corresp[api_key]
    title_csv = os.path.join("saved_results_scholar", folder, "csv_save", f"scholar_results_{cpt:03}{str_end}.csv")
    title_pickle = os.path.join("saved_results_scholar", folder, "pickle_save", f"scholar_results_{cpt:03}{str_end}.pkl")

    #data_df = pd.DataFrame()
    data_list = organic_results(s, api_key, start_page)
    print("waiting for organic results to save..")

    if data_list == "no_results":
      data_df = pd.DataFrame()
      title_csv = os.path.join("saved_results_scholar", folder, "csv_save", f"scholar_results_{cpt:03}{str_end}_no_results.csv")
      title_pickle = os.path.join("saved_results_scholar", folder, "pickle_save", f"scholar_results_{cpt:03}{str_end}_no_results.pkl")
    else:
      data_df = pd.DataFrame(data=data_list)

    data_df.to_csv(title_csv, encoding="utf-8", index=False)
    data_df.to_pickle(title_pickle)