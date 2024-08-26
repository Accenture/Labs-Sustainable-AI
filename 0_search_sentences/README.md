
# Building the search sentences

Here, we build the search sentences for different data sources.

## Usage

**Step 1:** Choose your key-words and place them in a csv file within the folder ``key_words``. Here, we consider three groups of key-words, called "Population", "Intervention_1" and "Intervention_2", and our csv file ``key-words-scholar.csv`` thus contains three columns. 

**Step 2:** Uncomment one (or several) of the functions corresponding to the data source for which you want to build search sentences. The choices are:
- ``scholar_build(full_sentence)`` for Google Scholar, where we set ``full_sentence`` to ``False`` if we want the program to divide the search sentence in mulitple search sentences so as to comply with the constraints of the Google Scholar search engine (maximum number of characters, maximum number of results available), or we set ``full_sentence`` to ``True`` if we want a single sentence
- ``ieee_build()`` for the IEEE Digital Library
- ``acm_build()`` for the ACM Digital Library
- ``arxiv_build()`` for arXiv
- ``paper_build()`` for a basic search sentence, not meant for a particular data source.

**Step 3:** Run the main script as follows:
```Shellsession
demo> python build_search_sentence.py
```


## Remark

For a different application, for instance with different groups of key-words, the script ``build_search_sentence.py`` would need to be adapted.