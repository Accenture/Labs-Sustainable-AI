# Initial Search

Here, we provide scripts for the initial search on Google Scholar.

## Usage

**Step 1:** Make an account on the serpAPI website (https://serpapi.com) to obtain a private API Key with associated tokens. A free plan sith serpAPI has 100 associated tokens per month. One token corresponds to 1 google scholar page (hence, several tokens are needed for a search with several associated pages).

**Step 2:** Copy your API Key in ``serpapi_GoogleSearch.py`` (the location is indicated by comments), and indicated which of the google scholar search sentences you want to query (with the variables `start_sentence` and `start_sentence`).

**Step 3:** To make the search through SerpAPI for Scholar, run:
```Shellsession
demo> python serpapi_GoogleSearch.py
```
and to do a first post-processing of the Scholar results, run:
```Shellsession
demo> python scholar_post_process_1.py
```