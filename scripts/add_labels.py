__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2024, Aalto Speech Research"

# Adding labels to log and test data tables.
# This version does not include gesture due to concern in recognition accuracy. 
# In cases were separation between prediction and gesture is difficult, ite labeled as UNKNOWN.

# NOTE: English typing log file is large. 
#       It is recommended to split file into parts and label each separately.

# Load libraries
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import math
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import difflib
import pickle
from itertools import islice
from collections import Counter
import sys
import time

############################
########### FUNCTIONS ##########
## Function for string comparison

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
# todo: check that the returned words are correct ones
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




############################
########## DATASETS ##########

## Load datasets
# ENGLISH
participants = pd.read_csv('data/processed2020/english/mobile_participants_v2.csv',na_values=['N'])
participants = participants[pd.notnull(participants["BROWSER"])]
sentences = pd.read_csv('data/processed2019/english/sentences.csv', sep = ",")
test_sections = pd.read_csv('data/processed2020/english/test_sections_mobile_v2.csv', low_memory=False)

#log_filename = sys.argv[0]
#log_filename = 'data/processed2020/english/split_data/log_data_0.csv'
#log_data = pd.read_csv(log_filename, sep = ",", low_memory=False)


#log_data = pd.read_csv('data/processed2020/english/log_data_filtered_mobile.csv', low_memory=False)
log_data = pd.read_csv('data/processed2020/english/log_data_filtered_mobile_clean.csv', nrows=10000)
log_filename = 'temp'



'''
# FINNISH
participants = pd.read_csv('data/processed2020/finnish/june/mobile_participants_v2.csv')
test_sections = pd.read_csv('data/processed2020/finnish/june/test_sections_mobile_v2.csv')
sentences = pd.read_csv('data/kirjoitustesti_csv_2020-06-03/kirjoitustesti_sentences_2020-06-03.csv', sep = ",")

#log_data = pd.read_csv('data/processed2020/finnish/june/log_data_filtered_mobile.csv', low_memory=False)
log_data = pd.read_csv('data/processed2020/finnish/june/log_data_filtered_mobile.csv', nrows=100000, low_memory=False)
# Fix names KEY ==> EVENT_KEY
log_data = log_data.rename(columns={'EVENT_KEY': 'KEY'})
log_data = log_data.rename(columns={'INPUT_TEXT': 'INPUT'})
log_data = log_data.rename(columns={'EVENT_TIMESTAMP': 'TIMESTAMP'})
'''

# Sorttaa log filu test_id, timestamp
log_data = log_data.sort_values(['TEST_SECTION_ID', 'TIMESTAMP'])



#########################
########## LABELS ##########

test_sections_ids = log_data['TEST_SECTION_ID'].unique()
test_sections = test_sections.loc[test_sections.TEST_SECTION_ID.isin(test_sections_ids)]

# todo: compute values for one test section id: 38661

keystrokess4ts = [[]]*test_sections.shape[0] # number of keystrokes
bs4ts = [np.nan]*test_sections.shape[0] # number of backspaces
ac4ts = [0]*test_sections.shape[0] # AC occurred
ac_e4ts = [0]*test_sections.shape[0] # Errors made by AC
ac_c4ts = [0]*test_sections.shape[0] # AC corrected by user
pred4ts = [0]*test_sections.shape[0] # prediction occured 
pred_e4ts = [0]*test_sections.shape[0] # errors made by prediction
pred_c4ts = [0]*test_sections.shape[0] # prediction corrected by user




first_ts = True
idx_counter = 0
for index, test_section in islice(test_sections.iterrows(), 0, None):

    # DEBUGLINE
    #if test_section.TEST_SECTION_ID != 4312:
    #    continue

    start = time.time()
    # Original sentence
    sentence_original = sentences[sentences.SENTENCE_ID == test_section.SENTENCE_ID].SENTENCE.tolist()[0]
    
    # Select logs
    log_subset = log_data.loc[log_data.TEST_SECTION_ID == test_section.TEST_SECTION_ID]

    # Save values: state in each
    input_list = ['']*log_subset.shape[0] 
    input_added = ['']*log_subset.shape[0]
    input_deleded = ['']*log_subset.shape[0]
    keystrokes_num = 0 # Number of keystrokes in test section


    new_sentence = True
    typo_list = []
    for i, (index_log, row_log) in enumerate(log_subset.iterrows()):


        # Skip noisy start
        if row_log.KEY == 'Enter': break
        if new_sentence and row_log.KEY == 'Space':
            continue

        current_input = str(row_log.INPUT)
        timestamp = row_log.TIMESTAMP

        #### START OF SENTENCE
        if new_sentence: 
            previous_input = ''
            timestamp_prev = row_log.TIMESTAMP
            word_idx = 0
            #word_idx_prev = -1
            word_list = [0]*len(sentence_original.split())
            word_list_AC = [0]*len(sentence_original.split())
            word_list_PR = [0]*len(sentence_original.split())
            new_sentence = False
            last_char = ''
            

        if row_log.KEY == 'Shift':
            input_list[i] = 'SHIFT'
            continue

        # Input field null
        if pd.isnull(current_input):
            continue
            
        # If NaN, skip
        if current_input == 'nan':
            current_input = ''

        # If input field is empty, skip
        if current_input == "" and len(previous_input)==1:
            input_list[i] = 'BACKSPACE'
            previous_input = current_input
            continue

        # Only space
        if current_input == ' ' and len(previous_input)==0:
            input_list[i] = 'SHIFT'
            previous_input = current_input
            continue
        if current_input == ' ' and len(previous_input)>1:
            input_list[i] = 'BACKSPACE'
            previous_input = current_input
            continue

        # in log data, the first letter does not sometimes appear even if key is pressed.
        if len(current_input) == 2 and len(previous_input)==0:
            keystrokes_num += 2
            previous_input = current_input
            continue

        #### ADDED AND DELETED CHARACTERS
        char_del, char_add = string_difference(previous_input, current_input)

        # Nothing has changed
        if len(char_del) == 0 and len(char_add) == 0:
            previous_input = current_input
            continue
        
        
        # KEYSTROKES
        keystrokes_num += 1

            
        #######################
        string_added = ''
        for char in char_add:
            string_added += char[0]
        input_added[i] = string_added

        string_deleded = ''
        for char in char_del:
            string_deleded += char[0]
        input_deleded[i] = string_deleded

        if string_added == ' ' and string_deleded == '':
            input_list[i] = 'SPACE'
            previous_input = current_input
            continue
            
        # Multiple words were deleted when sentence had been completed
        # todo: prosentti lauseen pituudesta
        # previous_input
        if len(string_deleded) > 20 and len(sentence_original)-len(previous_input) < 3:
            #current_input = previous_input
            input_list[i] = 'END'
            break



        #### ITE #####

        # More than one character was changed or inputted.
        if len(char_add)+len(char_del) > 1 :

            #if i> 0 and input_list[i-1] == 'ITE-correction':
            #    input_list[i] = ''

            if len(char_add) == 2 and char_add[0][0] == ' ' and len(char_del) == 0 and char_add[1][0].isalpha():
                input_list[i] = 'UNKNOWN'

            #if (string_added == ', ' or '. ') and string_deleded == '':
            elif len(char_add) == 2 and (char_add[0][0] == ',' or char_add[0][0] == '.' or char_add[0][0] == '?' or char_add[0][0] == '!') and char_add[1][0] == ' ' and len(char_del) == 0:
                input_list[i] = 'AC-COMMA'
                if input_list[i-1] == 'BACKSPACE':
                    input_list[i-1] = 'AUTO-INPUT'
            
            # GESTURE
            ## Whole word without space at the end word_idx_prev
            # First character is space (at least in some version)
            # Sometimes gesture starts with space.
            # Gesture can include whole word or part of the word (e.g. 3+ letters)
            elif len(char_add) > 2 and (char_add[-1][0] != ' ') and (input_list[i-1] == 'SPACE' or last_char == ' ' or i == 0 or char_add[0][0] == ' '):
            	#pass

                # Added chars are consecutive

                if i > 2 and char_add[0][0] == ' ' and input_list[i-2] == 'PREDICTION':
                    input_list[i] = 'PREDICTION'
                else:
                    input_list[i] = 'UNKNOWN' # not working well.
                if input_list[i-1] == 'MULT-CHAR-DEL' and input_list[i-3] == 'AC':
                    input_list[i] = 'ITE-CORRECTION'

                #input_list[i] = 'PREDICTION'
            
            # PREDICTION
                
            ## last character is not whitespace or dot
            # len(char_del) == 0 and
            elif len(char_add) > 1 and (char_add[-1][0] != ' ') and (char_add[0][0] != ' ') and (char_add[-1][0] != '.')  and (char_add[-1][0] != ',') and (char_add[-1][0] != ':') and (char_add[-1][0] != '?') and (char_add[-1][0] != '!'):
                if char_add[-2][0] == ' ' :
                    input_list[i] = 'PREDICTION'
                    previous_input = current_input
                    timestamp_prev = timestamp
                    continue
                else:
                    input_list[i] = 'PREDICTION'
                if timestamp-timestamp_prev < 30. and i != 0: # word disappears before being replaced by selected word
                    input_list[i-1] = 'AUTO-INPUT'
                    #if input_list[i-3] == 'AC':
                    #    input_list[i] = 'AC-CORRECTION'
                    if input_list[i-2] == 'BACKSPACE' or input_list[i-3] == 'AC':
                        input_list[i] = 'ITE-CORRECTION'
                    else:
                        input_list[i] = 'PREDICTION'


            # Whole word or multiple letters were deleted (AUTOCORRECTION CORRECTION)
            elif len(char_add) == 0:
                if (input_list[i-1] == 'SPACE' and input_list[i-2] == 'SPACE'):
                    input_list[i] = 'REPLACE-DOT'
                else: input_list[i] = 'MULT-CHAR-DEL'

            elif (input_list[i-1] == 'BACKSPACE' and input_list[i-2] == 'SPACE') and string_added == '. ':
                input_list[i] = 'REPLACE-DOT'
                if input_list[i-1] == 'BACKSPACE': 
                    input_list[i-1] = 'SPACE'

            # AUTOCORRECTION
            # length difference between previous and current input
            # note: no difference between auto-correction and auto-fill
            elif len(char_add) > 1 and input_list[i-1] != 'ITE-CORRECTION' and input_list[i-1] != 'REPLACE-DOT': 
                if char_add[-1][0] == ' ' or char_add[-1][0] == '.' or char_add[-1][0] == ',' or char_add[-1][0] == ':' or char_add[-1][0] == '?' or char_add[-1][0] == '!':
                    input_list[i] = 'AC'

                    if input_list[i-1] == 'BACKSPACE' and 'AC' in input_list: # AC is corrected
                        input_list[i] = 'ITE-CORRECTION'


        #### DELETED CHARACTERS
        # Backspace
        elif len(char_del) == 1 and len(char_add) == 0:
            input_list[i] = 'BACKSPACE'
            last_char = ''
            pass


        #### Last char in the sentence/log
        if i == len(log_subset) - 1:
            new_sentence = True
            break



        if len(char_add) > 0: 
            last_char = char_add[-1][0]
        else: 
            char_add = ''
        previous_input = current_input
        timestamp_prev = timestamp
        #word_idx_prev = word_idx



    #########################
    # ITE ACCURACY
    previous_input = ''
    for i, (index_log, row_log) in enumerate(log_subset.iterrows()):
        current_input = str(row_log.INPUT)
        #print(current_input)

        #### ADDED AND DELETED CHARACTERS
        char_del, char_add = string_difference(previous_input, current_input)

        # Nothing has changed
        if len(char_del) == 0 and len(char_add) == 0:
            previous_input = current_input
            continue

        # Current word
        ### LOCATION
        word_idx_new, current_word, original_word = locate_position_in_string(char_add, char_del, current_input, sentence_original)
        if word_idx_new != False:
            word_idx = word_idx_new
        # filter dots, commas etc from the words
        punc = '''!()-[]{};:"\,<>./?@#$%^&*_~'''
        for ele in current_word:
            if ele in punc:
                current_word = current_word.replace(ele, "")
        for ele in original_word:
            if ele in punc:
                original_word = original_word.replace(ele, "")


        # AC used for word
        if input_list[i] == 'AC':
            word_list_AC[word_idx] = 1

            # Accuracy: Error in AC
            if current_word != original_word:
                ac_e4ts[idx_counter] += 1 # ac error

        # PREDICTION used for word
        if input_list[i] == 'PREDICTION':
            word_list_PR[word_idx] = 1 

            # Accuracy: Error in Prediction
            if current_word != original_word:
                pred_e4ts[idx_counter] += 1 

                #print('error in prediction')
                #print(current_word+' '+original_word)


        # ITE corrected by user
        if input_list[i] == 'BACKSPACE' or input_list[i] == 'ITE-CORRECTION' or input_list[i] == 'MULT-CHAR-DEL':
            if word_list_AC[word_idx] == 1:
                ac_c4ts[idx_counter] += 1
                word_list_AC[word_idx] = 0

            if word_list_PR[word_idx] == 1:
                pred_c4ts[idx_counter] += 1
                word_list_PR[word_idx] = 0
                #print('corrected by user')

        # Remove auto-inputs from total keystroke number
        if input_list[i] == 'AUTO-INPUT':
            keystrokes_num -= 1

        previous_input = current_input

    

    # ADD VALUES TO LOG TABLE
    log_subset['INPUT_ADDED'] = input_added
    log_subset['INPUT_DELEDED'] = input_deleded
    log_subset['OBSERVED_KEY'] = input_list
    
    if first_ts:
        log_data_labeled = log_subset
        first_ts = False
    else:
        log_data_labeled = pd.concat([log_data_labeled, log_subset])
        
        

    # ADD VALUES TO TEST SECTION
    # Keystrokes
    keystrokess4ts[idx_counter] = keystrokes_num
    
    # Backspaces and ITE
    bs4ts[idx_counter] = Counter(input_list)['BACKSPACE']
    ac4ts[idx_counter] = Counter(input_list)['AC']
    pred4ts[idx_counter] = Counter(input_list)['PREDICTION']
        
    idx_counter += 1




test_sections['KEYSTROKES'] = keystrokess4ts
test_sections['BACKSPACES'] = bs4ts
test_sections['AC'] = ac4ts
test_sections['AC_ERROR'] = ac_e4ts
test_sections['AC_CORRECTION'] = ac_c4ts
test_sections['PREDICTION'] = pred4ts
test_sections['PREDICTION_ERROR'] = pred_e4ts
test_sections['PREDICTION_CORRECTION'] = pred_c4ts




log_data_labeled.to_csv('temp_log.csv', index=False)
test_sections.to_csv('temp_ts.csv', index=False)





###################################
################## SAVE ##########

#batch_num = log_filename.split('_')[-1].split('.')[0]
#new_log_filename = 'data/processed2020/english/split_data/log_data_labeled'+str(batch_num)+'.csv'
#new_ts_filename = 'data/processed2020/english/split_data/test_sections_labeled'+str(batch_num)+'.csv'
#new_p_filename = 'data/processed2020/english/split_data/participants_labeled'+str(batch_num)+'.csv'


#log_data_labeled.to_csv(new_log_filename, index=False)
#test_sections.to_csv(new_ts_filename, index=False)
#participants.to_csv(new_p_filename, index=False)


#log_data_labeled.to_csv('temp_log2.csv', index=False)



