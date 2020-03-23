
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
namelist = ["LinearRegression","LassoRegression","RandomForest"]
def Lasso_visualizationGraph(name):
    Lasso_data0 = pd.read_csv(os.getcwd() + "/../test/99_test_result_"+name+"_D0.csv", index_col=0)
    Lasso_data7 = pd.read_csv(os.getcwd() + "/../test/99_test_result_"+name+"_D7.csv", index_col=0)
    Lasso_data14 = pd.read_csv(os.getcwd() + "/../test/99_test_result_"+name+"_D14.csv", index_col=0)
    Lasso_data28 = pd.read_csv(os.getcwd() + "/../test/99_test_result_"+name+"_D28.csv", index_col=0)
    return Lasso_data0, Lasso_data7, Lasso_data14, Lasso_data28
def RF_visualizationGraph(name):
    RF_data0 = pd.read_csv(os.getcwd() + "/../test/99_test_result_" + name + "_D0.csv", index_col=0)
    RF_data7 = pd.read_csv(os.getcwd() + "/../test/99_test_result_" + name + "_D7.csv", index_col=0)
    RF_data14 = pd.read_csv(os.getcwd() + "/../test/99_test_result_" + name + "_D14.csv", index_col=0)
    RF_data28 = pd.read_csv(os.getcwd() + "/../test/99_test_result_" + name + "_D28.csv", index_col=0)
    return RF_data0, RF_data7, RF_data14, RF_data28

# Lasso_data0, Lasso_data7, Lasso_data14, Lasso_data28 =Lasso_visualizationGraph("LassoRegression")
# RF_data0, RF_data7, RF_data14, RF_data28 = RF_visualizationGraph("RandomForest")

def compareIndex(lasso, rf, day):
    df = pd.DataFrame(columns=("prediction","realValue","accuracy"))
    avr = 0;
    for i in range(len(lasso)):
        if (lasso.ix[i, 3] > rf.ix[i, 3]):
            df.loc[i] = [lasso.ix[i, 1],lasso.ix[i, 2],lasso.ix[i, 3]]
        else:
            df.loc[i] = [rf.ix[i, 1],rf.ix[i, 2],rf.ix[i, 3]]
        avr += df.ix[i, 2]
    avr /= len(lasso)
    df.to_csv(os.getcwd() + "/../test/d"+str(day)+".csv", encoding="utf-8")
    return df, avr


# df0, avr0 = compareIndex(Lasso_data0, RF_data0, 0)
# df7, avr7 = compareIndex(Lasso_data7, RF_data7, 7)
# df14, avr14 = compareIndex(Lasso_data14, RF_data14, 14)
# df28, avr28 = compareIndex(Lasso_data28, RF_data28, 28)
#
# print(df0)
# print("-"*80)
# print(df7)
# print("-"*80)
# print(df14)
# print("-"*80)
# print(df28)
# print("-"*80)
# print(avr0)
# print(avr7)
# print(avr14)
# print(avr28)


data0 = pd.read_csv(os.getcwd() + "/../test/d0.csv", index_col=0)
data7 = pd.read_csv(os.getcwd() + "/../test/d7.csv", index_col=0)
data14 = pd.read_csv(os.getcwd() + "/../test/d14.csv", index_col=0)
data28 = pd.read_csv(os.getcwd() + "/../test/d28.csv", index_col=0)

def drawPlt(data0, data7, data14, data28, index):
    real_value = [data0["realValue"][index],
                  data0["prediction"][index],
                  data7["prediction"][index],
                  data14["prediction"][index],
                  data28["prediction"][index]]
    prediction_model = [1, 2, 3, 4, 5]
    plt.figure(figsize=(10, 8))
    red_patch = mpatches.Patch(color="red", label="Final Audience")
    blue_patch = mpatches.Patch(color="green", label="Prediction Value")

    plt.legend(handles=[red_patch, blue_patch], loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)
    width = 0.35
    barlist = plt.bar(prediction_model, real_value, width, alpha = 0.9)
    barlist[0].set_color('r')
    for i in range(1,len(barlist)):
        barlist[i].set_color('g')

    plt.ylim(min(real_value) - min(real_value) / 3, max(real_value) + min(real_value) / 4)

    plt.xlim(0, 6)
    plt.xlabel('Model', position=(1, 0), fontsize=15, verticalalignment='center', horizontalalignment='center')
    plt.ylabel('Audience', position=(0, 1.02), fontsize=15, verticalalignment='bottom', horizontalalignment='left',
               rotation='horizontal')
    plt.yticks([], fontsize=12)
    plt.title('Prediction for Movie Audience', fontsize=20, verticalalignment='bottom', position=(0.5, 1.02))
    plt.xticks(prediction_model, ("Actual_Value", "D0", "D7", "D14", "D28"), fontsize=12)

    for a, b in zip(prediction_model, real_value):
        a -= 0.25
        val = int(b)
        b += (b / 100)
        plt.text(a, b, str(val))

    fig = plt.gcf()
    fig.savefig(os.getcwd()+'/../test/graph/plt_'+str(index)+'.png')
    plt.close()

ALL = len(data0)
for j in range(0,ALL):
    drawPlt(data0, data7, data14, data28, j)


# date = 0
# data = pd.read_csv(os.getcwd() + "/../data/99_output_LassoRegression_D" + date + ".csv", index_col=0)
# index = 0
# real_data= pd.read_csv(os.getcwd() + "/../data/99_output_LassoRegression_D0.csv", index_col=0)












# plt.show()


