

import pandas as pd

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
a=pd.read_csv('baseline.csv.gz')
a[list_classes]=a[list_classes]**1.3
a.to_csv("test3.csv.gz", index=False, compression='gzip')