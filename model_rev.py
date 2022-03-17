import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
import pickle

def mean_absolute_percentage_error(y_true, y_pred):
    """
    This function calculates mean_absolute_percentage_error

    Input:
        y_true (array): True values
        y_pred (array): Preicted values

    Returns:
        Calculated MAPE

    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
df = pd.read_excel('macrotrends_amazon.xlsx')

quater=pd.date_range('2005-03-31',periods=67,freq='Q')
df =df.set_index(quater[::-1])
df=df.drop('Quaterly',axis=1)

new_df=df.iloc[6:]
df_rev=pd.DataFrame()
df_rev['Revenue']=new_df['Revenue'][::-1]
df_rev=df_rev.set_index(new_df.index[::-1])

train_data = df_rev.iloc[:56]
test_data = df_rev.iloc[56:]



start = len(train_data)
end = len(train_data)+len(test_data)-1
mape=[]
rename = f'AR(1) Predictions'
for i in range(1,13):
    ARfit = AutoReg(train_data['Revenue'],lags=i).fit()

    #print(f'\nLag: {ARfit.k_ar}')
    #print(f'Coefficients:\n{ARfit.params}')
    predictions23 = ARfit.predict(start=start,end=end,dynamic=False).rename(rename)
    #print(mean_absolute_percentage_error(test_data[col],predictions23))
    mape.append([mean_absolute_percentage_error(test_data['Revenue'],predictions23),i])
print(min(mape))

start=len(df_rev)
end=len(df_rev)-1+5


ARfit = AutoReg(df_rev['Revenue'],lags=5).fit()
predictions23 = ARfit.predict(start=start,end=end,dynamic=False)



pickle.dump(ARfit, open('model_revenue.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_revenue.pkl','rb'))
print(model.predict(start=len(df_rev),end=len(df_rev)+5))