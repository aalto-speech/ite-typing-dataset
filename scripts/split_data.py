import pandas as pd
import numpy as np
import os

# Load data
participants = pd.read_csv('data/processed2020/english/mobile_participants_v2.csv',na_values=['N'], nrows=20)
test_sections = pd.read_csv('data/processed2020/english/test_sections_mobile_v2.csv', low_memory=False)

participants['PARTICIPANT_ID'] = participants['PARTICIPANT_ID'].astype('int')
test_sections['PARTICIPANT_ID'] = test_sections['PARTICIPANT_ID'].astype('int')
test_sections['TEST_SECTION_ID'] = test_sections['TEST_SECTION_ID'].astype('int')


participants_list = participants['PARTICIPANT_ID'].tolist()
p_lists = np.array_split(participants_list, 10)

print(p_lists)

for idx, p_list in enumerate(p_lists):
	print(p_list)
	ts_list = test_sections.loc[test_sections.PARTICIPANT_ID.isin(p_list)].TEST_SECTION_ID.tolist()
	print(ts_list)

	first_time = True
	for chunk in pd.read_csv('data/processed2020/english/log_data_filtered_mobile_clean.csv', chunksize=100000, low_memory=False):

		log_subset = chunk.loc[chunk.TEST_SECTION_ID.isin(ts_list)]
		log_subset = log_subset[['TEST_SECTION_ID', 'LOG_DATA_ID', 'TIMESTAMP', 'KEY', 'INPUT']]

		if first_time:
			file_name = 'data/processed2020/english/split_data/log_data_'+str(idx)+'.csv'
			log_subset.to_csv(file_name, index=False, header=True)
			first_time = False
		else:
			log_subset.to_csv(file_name, mode='a', index=False, header=False)

		break



