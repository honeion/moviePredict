import pandas as pd
import numpy as np
import os
name_list = ["LinearRegression_D0","LinearRegression_D7","LinearRegression_D14","LinearRegression_D28",
             "LassoRegression_D0","LassoRegression_D7","LassoRegression_D14","LassoRegression_D28",
             "RandomForest_D0","RandomForest_D7","RandomForest_D14","RandomForest_D28"]


def convert(data_name,name):
    data = pd.read_csv(os.getcwd() + "/../data/99_output_"+data_name+".csv", index_col=0)
    for i in range(0,len(data)):
        data.ix[i, "index"] = data.ix[i, "index"].replace("[","")
        data.ix[i, "index"] = data.ix[i, "index"].replace("]", "")
        data.ix[i, "realValue"] = data.ix[i, "realValue"].replace("[", "")
        data.ix[i, "realValue"] = data.ix[i, "realValue"].replace("]", "")
        data.ix[i, "accuracy"] = data.ix[i, "accuracy"].replace("[", "")
        data.ix[i, "accuracy"] = data.ix[i, "accuracy"].replace("]", "")
    data["index"] = data["index"].astype(int)
    data["prediction"] = data["prediction"].astype(int)
    data["realValue"] = data["realValue"].astype(int)
    data["accuracy"] = data["accuracy"].astype(float)
    data.to_csv(os.getcwd()+"/../data/99_result_"+name+".csv",encoding="utf-8")

for i in range(0,len(name_list)):
    convert(name_list[i],name_list[i])