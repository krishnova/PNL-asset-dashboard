'''
# For creating virtual env and installing neccesary lib
python -m venv myenv
myenv\scipts\activate
pip install fastapi
pip install uvicorn
uvicorn api.app:app --reload
'''
from fastapi import FastAPI
#from typing import Optional 

# loading trained forecast model 
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
import uvicorn
import math 
import json

app = FastAPI()

revenue_model = pickle.load(open('model_revenue.pkl','rb'))
cogs_model = pickle.load(open('model_cogs.pkl','rb'))
#rd_model = pickle.load(open('model_RD.pkl','rb'))
SGA_model = pickle.load(open('model_SGA.pkl','rb'))


@app.get("/forecast_revenue")
def return_forecast(quater : int):
    return {'Predictions':revenue_model.predict(start=61, end=61+quater)}

@app.get("/forecast_cogs")
def return_forecast(quater : int):
    return {'Predictions':cogs_model.predict(start=61, end=61+quater)}

@app.get("/forecast_SGA")
def return_forecast(quater : int):
    return {'Predictions':SGA_model.forecast(quater)}

# running the server
if __name__ == '__main__' : 
    uvicorn.run(app=app,host="127.0.0.1", port=8000, log_level="info")

#uvicorn api.app:app --reload