__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2024, Aalto Speech Research"

# Generate dictionary containing information on each word.


####### LIBRARIES #######
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import math
import os
import difflib
import pickle
from itertools import islice
from collections import Counter
import re



########################
####### FUNCTION #######
## Function for string comparison
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
            # todo: fix edit distance measure
            current_word = current_input.split()[w_idx]
            original_word = sentence_original.split()[w_idx]
                        
            char_add_w, char_del_w = string_difference(current_word, original_word[0:len(current_word)])
            edit_distance_c = len(char_add_w) + len(char_del_w)
            
            if edit_distance_c < len(current_word)*0.5 or edit_distance_c <= 2:
                return w_idx, current_word, original_word

            if w_idx < len(sentence_original.split())-2:
                original_word = sentence_original.split()[w_idx-1]
                char_add_w, char_del_w = string_difference(current_word, original_word[0:len(current_word)])
                edit_distance_p = len(char_add_w) + len(char_del_w)     
                
                if edit_distance_p < edit_distance_c and edit_distance_p < len(current_word)*0.5:
                    return w_idx-1, current_word, original_word
            
            if w_idx > 0 and w_idx < len(sentence_original.split())-1:
                original_word = sentence_original.split()[w_idx+1]
                char_add_w, char_del_w = string_difference(current_word, original_word[0:len(current_word)])
                edit_distance_n = len(char_add_w) + len(char_del_w)
                                
                if edit_distance_n < edit_distance_c and edit_distance_n < len(current_word)*0.5:
                    return w_idx+1, current_word, original_word
                  
            original_word = sentence_original.split()[w_idx]
            return w_idx, current_word, original_word
    
    return False, '', ''


# Generate dictionary containing information on each word.
def generate_dict2(test_sections, sentences, logs, ignorelist):

    word_dict = {}
    k=0

    # TEST SECTION
    for index, row_ts in islice(test_sections.iterrows(), 0, None):
        #debug
        #print(row_ts.TEST_SECTION_ID)
        
        if row_ts.TEST_SECTION_ID in ignorelist: continue

        sentence_original = sentences[sentences.SENTENCE_ID == row_ts.SENTENCE_ID].SENTENCE.tolist()[0]

        # todo: if edit distance is too large, skip
        char_add, char_del = string_difference(row_ts.USER_INPUT.strip(), sentence_original.strip())
        if len(char_add) > 15 or len(char_del) > 15:
            print('too many errors')
            continue

        #### LOG DATA FRAME
        df_log4ts = logs.loc[logs.TEST_SECTION_ID == row_ts.TEST_SECTION_ID]


        #### SAVE DETAILS
        word_times = [0]*len(sentence_original.split())
        word_backspaces = [0]*len(sentence_original.split())

        word_ed = [0]*len(sentence_original.split()) # edit distance

        word_AC = [0]*len(sentence_original.split())
        word_AC_err = [0]*len(sentence_original.split())
        
        word_pred = [0]*len(sentence_original.split())
        word_pred_err = [0]*len(sentence_original.split())
        
        word_keystrokes = [0]*len(sentence_original.split()) # for word, for sentence (test section)
        new_sentence = True


        ########################
        # LOGS
        for i, (index_log, row_log) in enumerate(df_log4ts.iterrows()):
                        
            if row_log.KEY == 'Enter': break
            if new_sentence and row_log.KEY == 'Space':
                continue

            current_input = str(row_log.INPUT)
            
            #### START OF SENTENCE
            if new_sentence:                 
                previous_input = ''
                previous_time = row_log.TIMESTAMP
                word_idx = 0
                word_keystrokes[word_idx] += 1
                new_sentence = False
                prev_ITE = ''
                continue
                
            if row_log.KEY == 'Shift':
                word_keystrokes[word_idx] += 1
                continue

            # Input field null
            if pd.isnull(current_input):
                continue
                
            # If NaN, skip
            if current_input == 'nan':
                current_input = ''

            # If input field is empty, skip
            if current_input == "" and len(previous_input)>0:
                word_backspaces[0] += 1
                word_keystrokes[0] += 1
                continue

            # Only space
            if current_input == ' ' and len(previous_input)==0:
                word_keystrokes[0] += 1
                continue
            if current_input == ' ' and len(previous_input)>1:
                word_backspaces[0] += 1
                word_keystrokes[0] += 1
                continue

            # needed if there is mysterious nans. however, removes space after word which is bad.
            #current_input = current_input.strip()


            #### ADDED AND DELETED CHARACTERS
            char_del, char_add = string_difference(previous_input, current_input)

            # Nothing has changed
            if len(char_del) == 0 and len(char_add) == 0:
                previous_input = current_input
                continue

            ### LOCATION
            word_idx_new = locate_position_in_string(char_add, char_del, current_input, sentence_original)
            if word_idx_new != False:
                word_idx = word_idx_new

            # Keystroke
            word_keystrokes[word_idx] += 1
            # Time
            word_times[word_idx] += row_log.TIMESTAMP - previous_time
            previous_time = row_log.TIMESTAMP


            #### BACKSPACE ####
            if row_log.OBSERVED_KEY == 'BACKSPACE':
                word_backspaces[word_idx] += 1
                
            #### ITE #####
            #######################  
            elif row_log.OBSERVED_KEY == 'AC':
                word_AC[word_idx] += 1
                prev_ITE = 'AC'
                
            elif row_log.OBSERVED_KEY == 'PRED':
                word_pred[word_idx] += 1
                prev_ITE = 'PREDICTION'
            
                
            elif row_log.OBSERVED_KEY == 'ITE-correction' and prev_ITE=='AC':
                word_AC_err[word_idx] += 1
                
            elif row_log.OBSERVED_KEY == 'ITE-correction' and prev_ITE=='PREDICTION':
                word_pred_err[word_idx] += 1


            #### Last char in the sentence/log
            if i == len(df_log4ts) - 1:
                new_sentence = True
                break

            previous_input = current_input
            



        #### ADD TO DICTIONARY
        for idx in range(len(word_times)):

            word = sentence_original.split()[idx]
            time_for_word = word_times[idx]

            #debug
            if time_for_word < 0:
                print(word, time_for_word)
                print(row_log)
                raise SystemExit("Negative word duration")
                break

            # initializing punctuations string
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for ele in word:
                if ele in punc:
                    word = word.replace(ele, "")
            # lowercase
            word = word.lower()


            # add word and and info
            if time_for_word != 0:
                ites = [word_AC[idx], word_AC_err[idx],word_pred[idx],word_pred_err[idx]]
                sample_info = [time_for_word, word_backspaces[idx], word_keystrokes[idx], word_ed[idx], ites]
                if word in word_dict: # word exists in dictionary
                    word_dict[word].append(sample_info)
                else:
                    word_dict[word] = [sample_info]
                    
            if row_ts.TEST_SECTION_ID > 50000*k:
                # Save word dictionary
                a_file = open("word_dict_temp.pkl", "wb")
                pickle.dump(word_dict, a_file)
                a_file.close()
                k += 1



    return word_dict


############################
####### LOAD DATASETS #######
#todo: change to labeled ones
df_sentences = pd.read_csv('data/processed2020/english/sentences.csv', sep = ",")
df_test_sections = pd.read_csv('data/processed2020/english/test_sections_labeled.csv', low_memory=False)
df_log_data = pd.read_csv('data/processed2020/english/log_data_labeled.csv')
df_log_data = df_log_data.sort_values(['TEST_SECTION_ID', 'TIMESTAMP'])


# Generate Dict
ignorelist=[]
word_dict_en = generate_dict2(df_test_sections, df_sentences, df_log_data, ignorelist)

# Save word dictionary
a_file = open("word_dict_temp.pkl", "wb")
pickle.dump(word_dict_en, a_file)
a_file.close()












