# Systematic Literature Review with Python


## Description
A Systematic Literature Review (SLR) is a form of secondary study that uses a well-defined methodology to identify, analyze and interpret all available evidence related to a specific research question in a way that is unbiased and (to a degree) repeatable. In contrast, the traditional literature review is a discursive text that may give subjective prominence to the most interesting or influential studies. The systematic review is an objective and standardised exercise. 

Within the AI Labs Sophia, and in collaboration with the INRIA Sophia, we use a SLR to answer the following question: _"What tools and methods currently permit to estimate the energy consumption of machine learning?"_

This repository contains several Python scripts and notebooks helping in this process. In particular, it permits to tackle the following problems:
- The methodology of an SLR entails using specific _key-words_ to build _query sentences_ for the chosen datasources (here, the Google Scholar search engine, ACM Digital Library and IEEE Digital Library). After having analyzed the datasources, we developed scripts to build appropriate query sentences for this data sources.
- Searches on Scholar present challenges. Indeed, a search sentence cannot exceed 256 characters, no search may yield more than 1000 results, and recovering results on the search engine is a slow process. Here, on the one hand, we have developed scripts to build appropriate search sentences to meet these constraints, and on the other hand, we have made these searches and recovered the results through the [SerpAPI](https://serpapi.com/google-scholar-api).
- Our research question is not precise and prone to leading to numerous unrelated results. We developed scripts to help in identifying (an thus removing) such results.
- More generally, we have developed scripts to analyse and postprocess the results at several stages of the SLR (such as identifying and removing duplicates, creating csv/excel tables interfaces to help analysis by hand of the results, creating LaTeX content).

## Repository organization

This repository is organized as follows:

- The folder ```0_search_sentences``` contains all files and scripts needed to to build search sentences for the different data sources (scholar, acm, ieee). In particular, the script ```build_search_sentences.py``` permits to create query sentences for several data sources: ACM, IEEE, arXiv, Google Scholar. The folder ```key_words``` contains our chosen key-words within a csv file. And the folders ```sentences_(...)``` contain the query sentences built for the datasources ACM, IEEE and Scholar.

- The folder ```1_initial_search``` contains all files and scripts concerning the initial search done on the three datasources. In particular, the script ```serpapi_GoogleSearch.py``` permits to make the Scholar queries and save the results directly via Python. It is based on the google API [SerpAPI](https://serpapi.com/google-scholar-api), and the tutorial on which our code is based can be found [here](https://serpapi.com/blog/scrape-historic-google-scholar-results-using-python/). It also contains the folders ```saved_results_(...)``` that gather the results of searches on the different datasources, as well as scripts to gather and explore these results (but not modify them). Note that one needs to create and account on SerpAPI to get a personal key and use the API.

- The folder ```2_automatic_selection``` contains all files and scripts needed to apply a semi-automatic selection on the search results obtained from the three data sources. More precisely, in the folder ```0_finding_excluding_words```, the notebook ```excluding_words.ipynb``` permits to build a list of words (among the words contained in all titles) that we consider automatically make a title containing one of these words off-topic: here we call these words 'excluding words'. The files ```all_words_2.csv``` and ```twoWords_20230705-115941.csv``` contain the words (or pair of words) of all titles in decreasing order of frequency; one of the columns contains a value for each word (entered by hand) indicating whether the word is an excluding word or not. The outcomes of applying the excluding words on the results of each of the 3 data sources is stored in the files ```ieee_df_20230718-173758.csv```, ```acm_df_20230718-172831.csv``` and ```scholar_keep_20230711-120831.csv```.

- The folder ```3_postprocess_automatic_selection``` contains the notebook ```chicago_post_process.ipynb``` that gathers the results of the automatic selection, recovers their bibliographic information (authors, journal or book, publication year, etc.), and prepares them in a format fit for doing a selection by hand. Here, we also use the raw results from serpAPI for Scholar, a copy of which is stored in the folder ```serpapi_cite_results```.

- The folder ```4_selection_by_hand``` contains the results of the selection by hand. The notebook ```1_load_selected_papers.ipynb``` load these results, and permit to create a new excel file fit to analyse the selected results. The latter are in fact classified into different groups, and the notebook ```2_load_classified_papers.ipynb``` permits to load each group separately and output their information in a format fit to prepare the data extraction form (the next step of the SLR).

- The folder ```5_data_extraction``` contains the results of the data extraction. The notebook ``load_data_extraction.ipnb`` loads these results and creates LaTeX tables and other contents.

- The folder ```6_comparison_surveys.ipynb``` contains the xslx files and a notebook ``comparison_surveys.ipynb`` permitting to observe what is the intersection between our survey and other surveys.


## Installation
Create a virtual environment and install the following packages with pip:
- google-search-results
- pandas
- xlsxwriter
- openpyxl
- numpy
- requests
- matplotlib
- seaborn

We have also installed ipykernel to run the jupyter notebooks with our virtual environment.

This code has mainly been developed with Python 3.9.13 (mac os) and Python 3.10.8 (windows). 



## Usage
To create the query sentences, within the first folder, run: 
```Shellsession
demo> python build_search_sentence.py
```
To make the search through SerpAPI for Scholar, within the second folder, run:
```Shellsession
demo> python serpapi_GoogleSearch.py
```
and do a first post-processing of the Scholar results:
```Shellsession
demo> python scholar_post_process_1.py
```
All other steps consist in running Jupyter Notebooks.

**All detail on the usage of this repository can be found in the README files or Jupyter Notebooks located in each subfolder.**