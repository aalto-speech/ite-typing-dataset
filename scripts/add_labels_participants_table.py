__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2019, Aalto Speech Research"

# Creating participants table with extral labels: observed ite, ite accuracy, kspc, backspace / char

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


###################################
########### LOAD DATA #############
#sentences = pd.read_csv('data/kirjoitustesti_csv_2020-06-03/kirjoitustesti_sentences_2020-06-03.csv')
#test_sections = pd.read_csv('data/processed2020/finnish/june/test_sections_labeled3.csv', low_memory=False)
#participants = pd.read_csv('data/processed2020/finnish/june/mobile_participants_v2.csv')

sentences = pd.read_csv('data/processed2019/english/sentences.csv', sep = ",")
participants = pd.read_csv('data/processed2020/english/mobile_participants_v2.csv',na_values=['N'])
test_sections = pd.read_csv('data/processed2020/english/test_sections_labeled3.csv', low_memory=False)
#test_sections = pd.read_csv('temp_ts.csv', low_memory=False)


###################################
########### PARTICIPANTS ##########

particpants_ids = test_sections['PARTICIPANT_ID'].unique()
participants = participants.loc[participants.PARTICIPANT_ID.isin(particpants_ids)]

# Looppaa test sectioneiden yli ja laske keskivarvot yms participants tauluun.

#ITE_list = ['']*participants.shape[0]
BSpC_list = [0]*participants.shape[0]
KSPC_list = [0]*participants.shape[0]
AC_list = [0]*participants.shape[0]
AC_error_list = [0]*participants.shape[0]
AC_correction_list = [0]*participants.shape[0]
PRED_list = [0]*participants.shape[0]
PRED_error_list = [0]*participants.shape[0]
PRED_correction_list = [0]*participants.shape[0]
#AC_acc_list = [None]*participants.shape[0] # AC accuracy
#pred_acc_list = [None]*participants.shape[0] # Prediction accuracy

idx_counter = 0
for id_p, participant in islice(participants.iterrows(), 0, None):

    ts_subset = test_sections.loc[test_sections.PARTICIPANT_ID == participant.PARTICIPANT_ID]
    
    BSPC4par = [0]*ts_subset.shape[0]
    KSPC4par = [0]*ts_subset.shape[0]
    AC4par = [0]*ts_subset.shape[0]
    AC_err4par = [0]*ts_subset.shape[0]
    AC_corr4par = [0]*ts_subset.shape[0]
    PRED4par = [0]*ts_subset.shape[0]
    PRED_err4par = [0]*ts_subset.shape[0]
    PRED_corr4par = [0]*ts_subset.shape[0]


    idx_ts = 0
    for id_ts, test_section in islice(ts_subset.iterrows(), 0, None):

        # number of characters
        sentence_original = sentences[sentences.SENTENCE_ID == test_section.SENTENCE_ID].SENTENCE.tolist()[0]
        num_chars = len(sentence_original)
        num_words = len(sentence_original.split(' '))

        BSPC4par[idx_ts] = test_section.BACKSPACES/num_chars
        KSPC4par[idx_ts] = test_section.KEYSTROKES/num_chars
        AC4par[idx_ts] = test_section.AC/num_words
        if test_section.AC > 0:
            AC_err4par[idx_ts] = test_section.AC_ERROR/test_section.AC
            AC_corr4par[idx_ts] = test_section.AC_CORRECTION/test_section.AC
        PRED4par[idx_ts] = test_section.PREDICTION/num_words
        if test_section.PREDICTION > 0:
            PRED_err4par[idx_ts] = test_section.PREDICTION_ERROR/test_section.PREDICTION
            PRED_corr4par[idx_ts] = test_section.PREDICTION_CORRECTION/test_section.PREDICTION
        idx_ts += 1

        
        
    # BACKSPACE ETC
    BSpC_list[idx_counter] = np.mean(BSPC4par)
    KSPC_list[idx_counter] = np.mean(KSPC4par)
    AC_list[idx_counter] = np.mean(AC4par)
    AC_error_list[idx_counter] = np.mean(AC_err4par) #ts_subset.AC_ERROR.sum()
    AC_correction_list[idx_counter] = np.mean(AC_corr4par) #ts_subset.AC_CORRECTION.sum()
    PRED_list[idx_counter] = np.mean(PRED4par)
    PRED_error_list[idx_counter] = np.mean(PRED_err4par) #ts_subset.PREDICTION_ERROR.sum()
    PRED_correction_list[idx_counter] = np.mean(PRED_corr4par) #ts_subset.PREDICTION_CORRECTION.sum()

    #AC_acc_list[idx_counter] = ts_subset['AC-ACCURACY'].mean()
    #pred_acc_list[idx_counter] = ts_subset['PRED-ACCURACY'].mean()
    idx_counter += 1

participants['BACKSPACES'] = BSpC_list
participants['KSPC'] = KSPC_list
participants['AC'] = AC_list
participants['AC_ERROR'] = AC_error_list
participants['AC_CORRECTION'] = AC_correction_list
participants['PREDICTION'] = PRED_list
participants['PREDICTION_ERROR'] = PRED_error_list
participants['PREDICTION_CORRECTION'] = PRED_correction_list

#participants['AC-ACCURACY'] = AC_acc_list
#participants['PRED-ACCURACY'] = pred_acc_list



###################################
################## SAVE ##########

#participants.to_csv('data/processed2020/finnish/june/participants_labeled3.csv', index=False)
participants.to_csv('data/processed2020/english/participants_labeled3.csv', index=False)
#participants.to_csv('temp_p.csv', index=False)

