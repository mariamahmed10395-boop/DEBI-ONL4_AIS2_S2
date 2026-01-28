import pandas as pd
def chk_type(df):
   dtypes=df.dtypes
   n_unique=df.nunique()
   print(pd.DataFrame({"Data":dtypes,"num_unique":n_unique}).T)
