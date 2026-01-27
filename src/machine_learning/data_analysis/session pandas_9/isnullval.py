def isNull(df):
    null=df.isnull().sum()
    return (pd.DataFrame(null))

isNull(df)