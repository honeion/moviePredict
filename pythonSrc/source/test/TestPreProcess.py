#-*- coding:utf-8 -*-
import pandas as pd
import os
modeling_data = pd.read_csv(os.getcwd() + '/../data/testData.csv', index_col=0)
# modeling_data = modeling_data[modeling_data["final_audience"] > 50000].reset_index(drop=True)
print(modeling_data)
# modeling_data.to_csv(os.getcwd()+"/../data/testData.csv", encoding="utf-8")

