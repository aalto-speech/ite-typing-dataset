# ITE typing dataset

Scripts and jupyter notebooks to process and analyse ITE typing dataset.


## Citation

Leino, Katri, Markku Laine, Mikko Kurimo, and Antti Oulasvirta. Mobile Typing with Intelligent Text Entry: A Large-Scale Dataset and Results.


## Jupyter Notebooks

notebooks/

* Typing_data_results.ipynb
  * Analysis on ITE and typing. File has all the results presented in the article.
* preprocessing_data_english.ipynb
  * Preprocessing English typing data. Filters out e.g. incomplete data.
* preprocessing_data_finnish.ipynb
  * Preprocessing Finnish typing data. Filters out e.g. incomplete data.


## Python scripts

scripts/

* add_labels.py
  * Adds ITE labels to log and test data tables.
* select_ite_words.py
  * Generates csv file with Autocorrected and SB selected words.
* add_labels_participants_table.py
  * Add ITE labels to participants table
* generate_dictionary.py
  * Generates dictionary file (word_dict3_en.pkl and word_dict3_fi.pkl)
* split_data.py
  * Splits log data into smaller tables


Scirpts used to select sentences for the typing test.
* scoring_sentences.py
* select_sentence


## Dataset:

data/

Dataset can be downloaded from Zenado: https://doi.org/10.5281/zenodo.12528163

Please extract data into files directory.

See data/README-datasets for more information.


## Files

Files can be downloaded from Zenado: https://doi.org/10.5281/zenodo.12528163

Please extract files into files directory.

files/

* vocab_fi_all_size237962101.pkl
  * The frequencies of the word in Finnish test sentences. Subset of Suomi24 and Finnish news corpora.
* vocab_giga_enron_size915074149.pkl
  * The frequencies of the word in English test sentences. Gigaword and Enron corpora used to caculate the frequencies.
* word_dict3_en.pkl
  * Contains information for each word e.g. the average typing time, number of BS/ITE used.
* word_dict3_fi.pkl
  * Contains information for each word e.g. the average typing time, number of BS/ITE used.


## Typing test

* kirjoitustesti-master.zip

    Compressed zip file contains typing speed test application for Finnish language. The source code is the updated version of the typing test application which has been previously used to collect large sets of observations for typing on a physical keyboard and on mobile devices.



## License

Distributed under the terms of the MIT license, see the LICENSE.txt file for details.

Copyright © 2024 Aalto Speech Recognition group, Aalto University, Finland


