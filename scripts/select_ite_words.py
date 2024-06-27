__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2024, Aalto Speech Research"

# Generate ac_words and pred_words tables.
# Code selects AC and SB words: 
# 1. the word typed by user before ITE is used, 
# 2. the word after ITE is used and
# 3. the original word user was supposed to type.

# Load libraries
import pandas as pd
import numpy as np
import math
import difflib
import pickle
from itertools import islice

# FUNCTIONS

def string_difference(current_input, previous_input):
    char_add = []
    char_del = []
    for i,s in enumerate(difflib.ndiff(current_input, previous_input)):
        if s[0]==' ': continue # char is the same in both
        elif s[0]=='-': # char has been added
            char_add.append([s[-1],i])
        elif s[0]=='+': # char has been deleted
            char_del.append([s[-1],i])
    return char_add, char_del


# EDIT DISTANCE
# Function to find Levenshtein distance between string `X` and `Y`.
# `m` and `n` is the total number of characters in `X` and `Y`, respectively
def edit_dist(X, m, Y, n):
 
    # base case: empty strings (case 1)
    if m == 0:
        return n
 
    if n == 0:
        return m
 
    # if the last characters of the strings match (case 2)
    cost = 0 if (X[m - 1] == Y[n - 1]) else 1
 
    return min(edit_dist(X, m - 1, Y, n) + 1,        # deletion (case 3a))
            edit_dist(X, m, Y, n - 1) + 1,           # insertion (case 3b))
            edit_dist(X, m - 1, Y, n - 1) + cost)    # substitution (case 2 + 3c)

# Current location in the string
def locate_position_in_string(char_add, char_del, current_input, sentence_original):
    current_input = current_input.lower()
    sentence_original = sentence_original.lower()
        
    if len(char_del) != 0:
        if char_del[0][1] > len(current_input)-1:
            last_letter_changed = len(current_input)-1
        else: last_letter_changed = char_del[0][1]
    elif len(char_add) != 0:
        if char_add[-1][1] > len(current_input)-1:
            last_letter_changed = len(current_input)-1
        else: last_letter_changed = char_add[-1][1]
    else:
        return False, '', ''

    if len(current_input.split()) == 0: 
        return False, '', ''
        
    char_count=0
    for w_idx, word in enumerate(sentence_original.split()):
        char_count += len(word)+1
        if char_count > last_letter_changed:
            
            if w_idx >= len(current_input.split()):
                #current_word = current_input.split()[-1]
                w_idx = len(current_input.split())-1
            #else: current_word = current_input.split()[w_idx]

            # in prediction and gesture input can be 'word n' 
            # where the last letter is the first letter of the next word.
            if len(char_add) > 2 and char_add[-2][0] == ' ' and w_idx > 0:
                w_idx -= 1
                current_word = current_input.split()[w_idx]
                original_word = sentence_original.split()[w_idx]
                return w_idx, current_word, original_word
            
            # Check edit distance of close words in case of typos etc.
            current_word = current_input.split()[w_idx]
            original_word = sentence_original.split()[w_idx]
                        
            char_add_w, char_del_w = string_difference(current_word, original_word[0:len(current_word)])
            edit_distance_c = len(char_add_w) + len(char_del_w)
            #edit_distance_c = edit_dist(current_word, len(current_word), original_word, len(original_word))
            
            if edit_distance_c < len(current_word)*0.5 or edit_distance_c <= 2:
                return w_idx, current_word, original_word

            if w_idx < len(sentence_original.split())-2:
                original_word = sentence_original.split()[w_idx-1]
                char_add_w, char_del_w = string_difference(current_word, original_word[0:len(current_word)])
                edit_distance_p = len(char_add_w) + len(char_del_w)
                #edit_distance_p = edit_dist(current_word, len(current_word), original_word[0:len(current_word)], len(original_word[0:len(current_word)]))

                
                if edit_distance_p < edit_distance_c and edit_distance_p < len(current_word)*0.5:
                    return w_idx-1, current_word, original_word
            
            if w_idx > 0 and w_idx < len(sentence_original.split())-1:
                original_word = sentence_original.split()[w_idx+1]
                char_add_w, char_del_w = string_difference(current_word, original_word[0:len(current_word)])
                edit_distance_n = len(char_add_w) + len(char_del_w)
                #edit_distance_n = edit_dist(current_word, len(current_word), original_word[0:len(current_word)], len(original_word[0:len(current_word)]))

                                
                if edit_distance_n < edit_distance_c and edit_distance_n < len(current_word)*0.5:
                    return w_idx+1, current_word, original_word
                  
            original_word = sentence_original.split()[w_idx]
            return w_idx, current_word, original_word
    
    return False, '', ''



def select_ite_words(log_data, test_sections_ac, sentences):

    typo_data_ac = []
    typo_data_pred = []

    for index, test_section in islice(test_sections_ac.iterrows(), 0, None):

    
        sentence_original = sentences[sentences.SENTENCE_ID == test_section.SENTENCE_ID].SENTENCE.tolist()[0]
        log_subset = log_data.loc[log_data.TEST_SECTION_ID == test_section.TEST_SECTION_ID]
        
        inputs = log_subset.INPUT.tolist()
        observed_keys = log_subset.OBSERVED_KEY.tolist()
        
        
        for i in range(1, len(inputs)-1):

            if observed_keys[i] == 'AUTO-INPUT': continue

            ite_word = ''

            if observed_keys[i-1] == 'AUTO-INPUT':
                current_input = str(inputs[i])
                previous_input = str(inputs[i-2])

                char_del, char_add = string_difference(previous_input, current_input)
                word_idx_new, current_word, original_word = locate_position_in_string(char_add, char_del, previous_input, sentence_original)

                word_idx_new, ite_word, original_word = locate_position_in_string(char_add, char_del, current_input, sentence_original)

            else: 
                current_input = str(inputs[i])
                previous_input = str(inputs[i-1])
            
                char_del, char_add = string_difference(previous_input, current_input)
                word_idx_new, current_word, original_word = locate_position_in_string(char_add, char_del, current_input, sentence_original)
                

            if observed_keys[i-1] != 'AUTO-INPUT' and (observed_keys[i+1] == 'AC' or observed_keys[i+1] == 'PREDICTION'):
                
                next_input = str(inputs[i+1])
                #ac_correted_word = next_input.split(' ')[word_idx_new]
                
                char_del, char_add = string_difference(current_input, next_input)
                word_idx_new, ite_word, original_word = locate_position_in_string(char_add, char_del, next_input, sentence_original)

                
            # # remove dots etc
            current_word = current_word.lower()
            ite_word = ite_word.lower()
            original_word = original_word.lower()
            
            current_word = current_word.replace('.', '')
            ite_word = ite_word.replace('.', '')
            original_word = original_word.replace('.', '')
            current_word = current_word.replace(',', '')
            ite_word = ite_word.replace(',', '')
            original_word = original_word.replace(',', '')
            current_word = current_word.replace('?', '')
            ite_word = ite_word.replace('?', '')
            original_word = original_word.replace('?', '')
            current_word = current_word.replace('!', '')
            ite_word = ite_word.replace('!', '')
            original_word = original_word.replace('!', '')

                
            if observed_keys[i+1] == 'AC' or (observed_keys[i] == 'AC' and observed_keys[i-1] == 'AUTO-INPUT'):
                typo_data_ac.append([test_section.TEST_SECTION_ID, test_section.SENTENCE_ID, current_input, current_word, ite_word, original_word])
            if observed_keys[i+1] == 'PREDICTION' or (observed_keys[i] == 'PREDICTION' and observed_keys[i-1] == 'AUTO-INPUT'):
                typo_data_pred.append([test_section.TEST_SECTION_ID, test_section.SENTENCE_ID, current_input, current_word, ite_word, original_word])

            
    df_typo_words_ac = pd.DataFrame(typo_data_ac, columns=['TEST_SECTION_ID', 'SENTENCE_ID', 'CURRENT_INPUT','TYPED_WORD', 'AC_WORD', 'ORIGINAL_WORD'])  
    df_typo_words_pred = pd.DataFrame(typo_data_pred, columns=['TEST_SECTION_ID', 'SENTENCE_ID', 'CURRENT_INPUT','TYPED_WORD', 'PRED_WORD', 'ORIGINAL_WORD'])  

    return df_typo_words_ac, df_typo_words_pred



# Data
# FI
'''
log_data_fi = pd.read_csv('data/processed2020/finnish/log_data_labeled.csv')
test_sections_fi = pd.read_csv('data/processed2020/finnish/test_sections_labeled.csv')
#test_sections_ac_fi = test_sections_fi.loc[test_sections_fi['AC'] > 0]
#ts_list = test_sections_ac_fi['TEST_SECTION_ID'].tolist()
#log_data_ac = log_data_fi[log_data_fi.TEST_SECTION_ID.isin(ts_list)]
sentences_fi = pd.read_csv('data/kirjoitustesti_csv_2020-06-03/kirjoitustesti_sentences_2020-06-03.csv', sep = ",")

# FINNISH
df_typo_words_ac, df_typo_words_pred = select_ite_words(log_data_fi, test_sections_fi, sentences_fi)
df_typo_words_ac.to_csv('ac_words_fi.csv', index=False)
df_typo_words_pred.to_csv('pred_words_fi.csv', index=False)

'''

# EN
# Note: remove nrows=5000 if intended to use whole file.
log_data_en = pd.read_csv('data/processed2020/english/log_data_labeled.csv',  nrows=5000)
test_sections_en = pd.read_csv('data/processed2020/english/test_sections_labeled.csv', low_memory=False,  nrows=100)
#test_sections_ac_en = test_sections_en.loc[test_sections_en['AC'] > 0]
#ts_list = test_sections_ac_en['TEST_SECTION_ID'].tolist()
#log_data_ac_en = log_data_en[log_data_en.TEST_SECTION_ID.isin(ts_list)]
sentences_en = pd.read_csv('data/processed2019/english/sentences.csv', sep = ",")


# ENGLISH
df_typo_words_ac, df_typo_words_pred = select_ite_words(log_data_en, test_sections_en, sentences_en)
df_typo_words_ac.to_csv('ac_words_en_temp.csv', index=False)
df_typo_words_pred.to_csv('pred_words_en_temp.csv', index=False)


