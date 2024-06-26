__author__ = "Katri Leino"
__copyright__ = "Copyright (c) 2019, Aalto Speech Research"

# Generating data set which ranks sentences according to difficulty
# Output file format: sentence_length , difficulty (int), sentence

import numpy as np
import math

# File
#f = open('/l/kkleino/LMKB/data/webcon_set2.txt','r') 
#f = open('/l/kkleino/LMKB/data/ylenews-fi-2011-2018-selko-src/yle_selkokieli_sentences_clean2.txt','r') 
f = open('/l/kkleino/data/suomi24/comments2013d.txt','r') 

###### VOCABULARY
vocabulary = dict()
vocabulary_size = 0

# Creating vocabulary with word frequencies
for line in f:
    wordlist = line.split()    

    for word in wordlist:
        if word in vocabulary:
            vocabulary[word] = vocabulary[word] + 1
        else:
            vocabulary[word] = 1
        vocabulary_size = vocabulary_size + 1

f.close()


###### SCORE

# File
f = open('/l/kkleino/data/suomi24/comments2013d.txt','r') 
f_out = open('data/suomi24_comments2013d_scored.txt', 'w')

for line in f:
    words = line.split()
    sentence_length = np.size(words)

    # Cleanup
    ## Empty line
    if sentence_length == 0: continue

    score = 0

    # Word based analysis on sentence difficulty
    for word in words:
        word_length = len(word)
        if word_length == 1 and word != 'o': 
            break
        #score += word_length*(1-vocabulary[word]*100/vocabulary_size)
        score += word_length*math.log(vocabulary[word]/vocabulary_size)
    score = score/sentence_length
    f_out.write(str(sentence_length)+"-"+str(abs(int(score/10)))+"-"+line)
    #f_out.write(str(int(score))+"-"+line)
    #print(str(score)+" "+str(sentence_length)+" "+line)



f.close()
f_out.close()



