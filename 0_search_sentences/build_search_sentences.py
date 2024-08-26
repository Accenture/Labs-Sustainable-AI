# Main script to build search sentences for different data sources
# Refer to README.md for more information on the usage of this script.



# Importing Libraries:
import pickle
import pandas as pd
import json
import os



# Usefull function:
def get_keywords(file_name):
    """ Loading the key-words stored within a csv file with three columns:
    Population, Intervention_1 and Intervention_2. """

    file = pd.read_csv(file_name,sep=';', header=0)
    df=pd.DataFrame(file)
    print(df)
    pop = [ val for val in df['Population'].values if val == val]
    int1 = [ val for val in df['Intervention_1'].values if val == val]
    int2 = [ val for val in df['Intervention_2'].values if val == val]
    exclude = [ val for val in df['Not'].values if val == val]

    return(pop, int1, int2, exclude)



# -------------------------- #
# ----- Google Scholar ----- #
# -------------------------- #

def scholar_sentence(*lists):
    """ Builds a search sentence for the Google Scholar search engine, 
    provided several lists of keywords."""

    sentence = ''

    def scholar_subsentence(words):
        subsentence = ''
        for elem in words:
            if ' ' in elem:
                subsentence += f'"{elem}"'
            else:
                subsentence += elem
            subsentence += '|'
        subsentence = subsentence[:-1]
        return(subsentence)

    for sublist in lists[:3]:
        subsentence = scholar_subsentence(sublist)
        sentence += f'({subsentence})'
        sentence += ' + '
    sentence = sentence[:-3]
    for word in lists[3]:
        sentence += ' -' + word
    sentence = 'allintitle:' + sentence

    return(sentence)


def scholar_build(full_sentence):
    """ If full_sentence == True -> Builds a single search sentence for Google Scholar.
    If full_sentence == False -> Builds several search sentences for Google Scholar, where
    each sentence has less than 256 characters. 
    The sentence/sentences is/are written in the file sentence-scholar.txt within the folder
    sentences_scholar """

    file_name = os.path.join("key_words",'key-words-scholar.csv')
    W1, W2, W3, exclude = get_keywords(file_name)

    if full_sentence:
        sentence, len_sentence = scholar_sentence(W1, W2, W3)
        with open("sentence-scholar-full.txt", "w") as f:
            f.write(sentence)
    else:
        # initialization of all variables:
        imax = len(W1); jmax = len(W2); kmax = len(W3)
        i = j = k = 0
        l1 = [W1[0]]; l2 = []; l3 = []            # will store words of each groups during the selection
        S = []                                    # will contain all selected sentences

        while i < imax:                           # loop on 1st list of keywords

            if j == jmax:                         # has already considered all words in W2 for current i - 1
                print(j==jmax, k==kmax)           # necessarily j==jmax and k==kmax

                S.append(current)                 # automatically store the previously accepted sentence
                print(len(current), current)
                print('x ---------- x')           # don't want to add a second word in l1
                print('x - stored - x')           # l1 only contains 1 word
                print('x ---------- x')
                l3 = []; k = 0                    # re-initialize l3 and k
                l2 = []; j = 0                    # re-initialize l2 and j
                l1 = [W1[i]]                      # consider next word in W1

            while j < jmax:                       # loop on 2nd list of keywords
                l2.append(W2[j])
                if k == kmax:                     # has already considered all words in W3 for current i & j - 1
                                                  # we check if we can add a new word from W2 in l2
                    s = scholar_sentence(l1, l2, l3, exclude)
                    # print(i, j, k); print(len(s), l1, l2, l3)
   
                    if len(s) <= 256:             # accept the new sentence
                        current = s
                    else:                         # don't accept the new sentence
                        S.append(current)         # store the previously accepted sentence
                        print(len(current), current)
                        print('x ---------- x')
                        print('x - stored - x')
                        print('x ---------- x')
                        l3 = []; k=0              # re-initialize l3 and k
                        l2 = [W2[j]]              # empty l2 except for the new word

                while k < kmax:                    # loop on 3rd list of keywords
                    l3.append(W3[k])
                    s = scholar_sentence(l1, l2, l3, exclude)
                    # print(i, j, k); print(len(s), l1, l2, l3)
                    if len(s) <= 256:              # accept the new sentence
                        current = s                
                    else:                          # don't accept the new sentence
                        S.append(current)          # store the previously accepted sentence
                        print(len(current), current)
                        print('x ---------- x')
                        print('x - stored - x')
                        print('x ---------- x')
                        l3 = []                    # empty l3, it will be filled by next word in k loop
                        
                    k+=1
                j+=1
            i+=1
        
        s = scholar_sentence(l1, l2, l3, exclude)  # the last sentence
        S.append(s)
        print("x ---------- x")
        print(len(S))  

        folder = "sentences_scholar"

        f_name = os.path.join(folder,"sentence-scholar.txt")
        with open(f_name, "w") as f:
            for s in S:
                # print(len(s), s)
                f.write(s)
                f.write("\n\n")

        f_name = os.path.join(folder,"sentence-scholar.json")
        S_dict = {}
        cpt = 1
        for s in S:
            S_dict[cpt] = s
            cpt+=1
        with open(f_name, "w") as f:
            json.dump(S_dict, f, indent=4, sort_keys=True)

        f_name = os.path.join(folder,'sentence-scholar.pkl')
        file = open(f_name, 'wb')
        # dump information to that file
        pickle.dump(S, file)
        # close the file
        file.close()



# ---------------- #
# ----- IEEE ----- #
# ---------------- #

def ieee_sentence(sublists):
    """ Builds a search sentence for the IEEE Digital Library, 
    provided several lists of keywords."""

    sentence = ""

    def ieee_subsentence(words):
        where = 'Document Title'
        subsentence = ""
        for elem in words:
            if ' ' in elem or '-' in elem:
                elem = f'"{elem}"'
            elem = f'"{where}":' + elem
            subsentence += elem
            subsentence += " OR "
        subsentence = subsentence[:-4]
        return(subsentence)
    
    for sublist in sublists[:-1]:
        subsentence = ieee_subsentence(sublist)
        sentence += f'({subsentence})'
        sentence += " AND "
    sentence = sentence[:-5]
    sentence += " NOT "
    sentence += f'({ieee_subsentence(sublists[-1])})'

    return(sentence)

def ieee_build():
    """ Builds a search sentence for the IEEE Digital Library.
    The sentence is written in the file sentence-ieee.txt within the folder
    sentences_other_datasources """

    file_name = os.path.join("key_words",'key-words-scholar.csv')
    # popu, inter, outc = get_keywords(file_name)
    W1, W2, W3, exclude = get_keywords(file_name)

    sentence = ieee_sentence([W1, W2, W3, exclude])
    
    file_name = os.path.join("sentences_other_datasources", "sentence-ieee.txt")
    with open(file_name, "w") as f:
        f.write(sentence)



# --------------- #
# ----- ACM ----- #
# --------------- #

def acm_sentence(sublists):
    """ Builds a search sentence for the ACM Digital Library, 
    provided several lists of keywords."""

    sentence = ""

    def acm_subsentence(words):
        subsentence = ""
        for elem in words:
            if ' ' in elem or '-' in elem:
                elem = f'"{elem}"'
            subsentence += elem
            subsentence += " OR "
        subsentence = subsentence[:-4]
        return(subsentence)

    for sublist in sublists[:-1]:
        subsentence = acm_subsentence(sublist)
        sentence += f'({subsentence})'
        sentence += " AND "
    sentence = sentence[:-5]
    # for word in sublists[-1]:
    #     sentence += " AND NOT "
    #     sentence += f'({word})'
    sentence += " NOT "
    sentence += f'({acm_subsentence(sublists[-1])})'

    return(sentence)

def acm_build():
    """ Builds a search sentence for the ACM Digital Library.
    The sentence is written in the file sentence-acm.txt within the folder
    sentences_other_datasources """

    file_name = os.path.join("key_words",'key-words-scholar.csv')
    W1, W2, W3, exclude = get_keywords(file_name)

    sentence = acm_sentence([W1, W2, W3, exclude])

    file_name = os.path.join("sentences_other_datasources", "sentence-acm.txt")
    with open(file_name, "w") as f:
        f.write(sentence)



# ----------------- #
# ----- arXiv ----- #
# ----------------- #

def arxiv_subsentence(words):
    subsentence = ""
    for elem in words:
     #   if ' ' in elem:
        elem = f'"{elem}"'
        subsentence += elem
        subsentence += " OR "
    subsentence = subsentence[:-4]
    print(subsentence)
    print(len(subsentence))
    return(subsentence, len(subsentence))

def arxiv_build():
    """ Builds a search sentence for arXiv, provided several lists of keywords.
    The sentence is written in the file sentence-arxiv.txt"""

    file_name = 'key-words.csv'
    popu, inter, outc = get_keywords(file_name)

    popu, len_popu = arxiv_subsentence(popu)
    inter, len_inter = arxiv_subsentence(inter)
    outc, len_outc = arxiv_subsentence(outc)

    with open("sentence-arxiv.txt", "w") as f:
        f.write(popu)
        f.write("\n\n")
        f.write(inter)
        f.write("\n\n")
        f.write(outc)



# ----------------- #
# ----- paper ----- #
# ----------------- #

def paper_subsentence(words):

    subsentence = ""
    for elem in words:
        subsentence += elem
        subsentence += ", "
    subsentence = subsentence[:-2]
    print(subsentence)
    print(len(subsentence))
    return(subsentence, len(subsentence))

def paper_build():
    """ Builds a search sentence not meant for a specific data source, provided several lists of keywords.
    The sentence is written in the file sentence-paper.txt"""

    file_name = 'key-words.csv'
    popu, inter, outc = get_keywords(file_name)

    popu, len_popu = paper_subsentence(popu)
    inter, len_inter = paper_subsentence(inter)
    outc, len_outc = paper_subsentence(outc)


    with open("sentence-paper.txt", "w") as f:
        f.write(popu)
        f.write("\n\n")
        f.write(inter)
        f.write("\n\n")
        f.write(outc)



if __name__ == '__main__':
    # uncomment bellow for the data source you are interested in:
    # scholar_build(False)
    # ieee_build()
    # acm_build()
    # arxiv_build()
    # paper_build()
    pass



# ---------------------- #
# -     APPENDIX       - #
# ---------------------- #
# -> Contains previous versions of the script for Google Scholar

# ----------------------------------- #
# -     First version Scholar       - #
# ----------------------------------- #
# def scholar_sentence(*subsentences):
#     sentence = ''
#     for subsentence in subsentences:
#         sentence += f'({subsentence})'
#         sentence += ' + '
#     sentence = sentence[:-3]
#     sentence = 'allintitle:' + sentence
#     print(sentence)
#     print(len(sentence))
#     return(sentence, len(sentence))
# def scholar_build(full_sentence):
#     file_name = 'key-words.csv'
#     popu, inter, outc = get_keywords(file_name)
#     if full_sentence:
#         popu, len_popu = scholar_subsentence(popu)
#         inter, len_inter = scholar_subsentence(inter)
#         outc, len_outc = scholar_subsentence(outc)
#         sentence, len_sentence = scholar_sentence(popu, inter, outc)
#         with open("sentence-scholar-full.txt", "w") as f:
#             f.write(sentence)
#     else:
#         inter1 = inter[:13]
#         inter2 = inter[13:]
#         inter, len_inter = scholar_subsentence(inter)
#         inter1, len_inter1 = scholar_subsentence(inter1)
#         inter2, len_inter2 = scholar_subsentence(inter2)
#         outc, len_outc = scholar_subsentence(outc)
#         with open("sentence-scholar.txt", "w") as f:
#             for word in popu:
#                 word, len_word = scholar_subsentence([word])
#                 sentence1, len_sentence1 = scholar_sentence(word, inter1, outc)
#                 f.write(sentence1)
#                 f.write("\n\n")
#                 sentence2, len_sentence2 = scholar_sentence(word, inter2, outc)
#                 f.write(sentence2)
#                 f.write("\n\n")
# ----------------------------------- #

# ----------------------------------- #
# - Works but not fofr what we want - #
# ----------------------------------- #
# imax = len(W1); jmax = len(W2); kmax = len(W3)
# print(imax, jmax, kmax)
# i = j = k = 0
# l1 = []; l2 = []; l3 = []
# # l1 = [W1[0]]; l2 = [W2[0]]; l3 = [W3[0]]
# S = []
# while i < imax:
#     l1.append(W1[i])
#     if j == jmax:
#         s = scholar_sentence(l1, l2, l3)
#         print(i, j, k)
#         print(len(s), l1, l2, l3)
#         if len(s) <= 256:
#             current = s
#         else:
#             print('" ---- stored -----" ')
#             S.append(current)
#             l3 = []
#             if k == kmax:
#                 l2 = []
#                 k = 0
#                 if j == jmax:
#                     l1 = [W1[i]]
#                     j = 0
#     while j < jmax:
#         l2.append(W2[j])
#         if k == kmax:
#             s = scholar_sentence(l1, l2, l3)
#             print(i, j, k)
#             print(len(s), l1, l2, l3)
#             if len(s) <= 256:
#                 current = s
#             else:
#                 print('" ---- stored -----" ')
#                 S.append(current)
#                 l3 = []
#                 if k == kmax:
#                     l2 = [W2[j]]
#                     k = 0
#                     # if j == jmax - 1:
#                     #     l1 = []
#                     #     j = -1
#         while k < kmax:
#             l3.append(W3[k])
#             s = scholar_sentence(l1, l2, l3)
#             print(i, j, k)
#             print(len(s), l1, l2, l3)
#             if len(s) <= 256:
#                 current = s
#             else:
#                 print('" ---- stored -----" ')
#                 S.append(current)
#                 l3 = []
#                 # if k == kmax - 1:
#                 #     l2 = []
#                 #     k = -1
#                     # if j == jmax - 1:
#                     #     l1 = []
#                     #     j = -1
#             k+=1
#         j+=1
#     i+=1
# # the last sentence:
# s = scholar_sentence(l1, l2, l3)
# S.append(s)
# print("------------")
# print(len(S))   
# with open("sentence-scholar.txt", "w") as f:
#     for s in S:
#         # print(len(s), s)
#         f.write(s)
#         f.write("\n\n")
# ----------------------------------- #