import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
import pickle
from statsmodels.tsa.holtwinters import ExponentialSmoothing


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

df['SG&A Expenses']=df['\xa0SG&A Expenses']

new_df=df.iloc[6:]

df_rev=pd.DataFrame()
df_rev['SG&A Expenses']=new_df['SG&A Expenses'][::-1]
df_rev=df_rev.set_index(new_df.index[::-1])

train_data = df_rev.iloc[:56]
test_data = df_rev.iloc[56:]

fitted_model1 = ExponentialSmoothing(train_data['SG&A Expenses'],trend='add',seasonal='add',seasonal_periods=4).fit()
test_predictions = fitted_model1.forecast(5).rename('HW Forecast')

print(mean_absolute_percentage_error(test_data['SG&A Expenses'], test_predictions))


fitted_model1 = ExponentialSmoothing(df_rev['SG&A Expenses'],trend='add',seasonal='add',seasonal_periods=4).fit()
predictions = fitted_model1.forecast(5).rename('HW Forecast')


import pickle

pickle.dump(fitted_model1, open('model_SGA.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_SGA.pkl','rb'))
print(model.forecast(5))