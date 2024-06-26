__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2019, Aalto Speech Research"

# Selects sentence from text file based on length and difficulty
# Data file format: sentence_length-difficulty(int)-sentence
# For further inquiries: katri.k.leino(a)aalto.fi

import random

# Consors sentences with listed words.
def censor(sentence):
    censored_words = ['vittu', 'vitun', 'perkele', 'helvetti', 'helvetisti', 'helvetin', 'jumalauta', 'panna', 'nussii', 'nussi', 'nussia', 'saatana', 'bitch', 'perse', 'vitut', 'huume', 'huumeet', 'paska', 'tissit', 'känni']
    abbreviation =['bee', 'cee', 'dee', 'ee', 'äf', 'gee', 'hoo', 'ii', 'jii', 'koo', 'äl', 'äm', 'än','pee','är','äs', 'vee', 'äks', 'yy']
    wordlist = sentence.split()
    for word in wordlist:
        if word in censored_words: return True
        elif word in abbreviation: return True
    return False


for sentence_length in range(3,10):

    # One file per length
    filename = 'len_'+str(sentence_length)
    f_out = open('data/kirjoitustesti/suomi24/'+filename+'.txt', 'w')

    for difficulty in range(4,11):
        ## PARAMETERS
        #sentence_length = sen_len
        #difficulty = diff
        print_n = 25

        # Data file
        f_in = open('data/suomi24_comments2013d_scored.txt', 'r')

        sentences = []
        for line in f_in:
            parts = line.split("-")
            if int(parts[0]) == sentence_length:
                if int(parts[1]) == difficulty:
                    sentence = '-'.join(parts[2:])
                    sentences.append(sentence.rstrip())
                    #if censor(parts[2].rstrip()): continue
                    #sentences.append(parts[2].rstrip())

        #filename = str(sentence_length)+'-'+str(difficulty)
        #f_out = open('data/ranked/'+filename+'.txt', 'w')
        for i in range(print_n):
            f_out.write(random.choice(sentences)+'\n')
        #f_out.close()

        f_in.close()





