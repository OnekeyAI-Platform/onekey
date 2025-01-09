#!/usr/bin/env python
# coding: utf-8


# Multi Instance Learning
# convert the patch-level feature to Histogram and TF-IDF feature
Referance: [Development and interpretation of a pathomics-based model for the prediction of microsatellite instability in Colorectal Cancer](http://www.medai.icu/download?url=http://www.medai.icu/apiv3/attachment.download?sign=1667478d908313ae1e01543e229d02de&attachmentsId=1061&threadId=230)



import pandas as pd
from onekey_algo.custom.utils import key2
import numpy as np
import os

log = pd.read_csv('your patch file path', sep='\t',
                 names=['fname', 'prob', 'pred', 'gt'])
log['prob'] = list(map(lambda x: x[0] if x[1] == 1 else 1-x[0], np.array(log[['prob', 'pred']])))
log[['group']] = log[['fname']].applymap(lambda x: os.path.basename(x).split('_')[0])
log['prob'] = log['prob'].round(decimals=2)
log.head()



# Histogram
# all data generate histogram features, multiple histo_columns are present, and all features are stitched horizontally
# group_column: column name of sample group, sample ID 
# histo_columns: the columns used to calculate the histogram, if there are multiple columns, each column calculates the histogram, and then the features are stitched       
# histo_lists: none or the same number as histo_columns, specifying a list of traits for yourself        
# default_value: default value when no feature exists

def key2histogram(data: pd.DataFrame, group_column: str, histo_columns: Union[str, List[str]],
                  histo_lists: Union[list, List[list]] = None, default_value=0, norm: bool = False):
results = key2.key2histogram(log, group_column='group',histo_columns='prob', norm=True)
results.to_csv('histogram.csv', header=True, index=False)
results




# Term Frequency-Inverse Document Frequency (TF-IDF)
# all data generate histogram features, multiple corpus_columns are present, and all features are stitched horizontally.  
# group_column: column name of sample group, sample ID 
# histo_columns: the column name used to calculate the corpus.
def key2tfidf(data: pd.DataFrame, group_column: str, corpus_columns: Union[str, List[str]]):
results = key2.key2tfidf(log, group_column='group',corpus_columns='prob')
results.to_csv('tfidf.csv', header=True, index=False)
results

