## Decision tree, Random forest and Logistic regression models
## for predicting Substance Abuse Recurrence
## Important features are also extracted

# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# # Reading csv file


pre_opioid = pd.read_csv('Opioid.csv')



pre_opioid.head()



pre_opioid['NOPRIOR']



pre_opioid.info()


# # Extracting key features


opioid = pre_opioid[['GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'VET', 'LIVARAG', 'PRIMINC', 'STFIPS', 'CBSA', 'PMSA', 'REGION', 'DIVISION','HLTHINS', 'PRIMPAY','NOPRIOR']].copy()



opioid


# # Removing rows where 'NOPRIOR' is not collected/missing


opioid = opioid[(opioid['NOPRIOR'] != -9)]
opioid['NOPRIOR']
opioid


# # Creating feature list


features = list(opioid)
features.remove('NOPRIOR')
features


# # Grouping multiple recurrences into 1 category


opioid.NOPRIOR[opioid.NOPRIOR == 1] = 1
opioid.NOPRIOR[opioid.NOPRIOR == 2] = 1
opioid.NOPRIOR[opioid.NOPRIOR == 3] = 1
opioid.NOPRIOR[opioid.NOPRIOR == 4] = 1
opioid.NOPRIOR[opioid.NOPRIOR == 5] = 1



opioid['NOPRIOR']
opioid['NOPRIOR'].value_counts()


# # Splitting data into training/testing sets


from sklearn.model_selection import train_test_split



X = opioid.drop('NOPRIOR', axis = 1)
y = opioid['NOPRIOR']



y



opioid['NOPRIOR'].value_counts()



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random
import time


# # Decision tree


dtree = DecisionTreeClassifier()



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)



start = time.time()
dtree.fit(X_train, y_train)
end = time.time()
time_dtree_fit = end - start
print("Time to create decision tree model: " + str(time_dtree_fit) + " seconds")



y_test.value_counts()



start = time.time()
predict_tree = dtree.predict(X_test)
end = time.time()
time_dtree_pred = end - start

print("Time to predict with decision tree model: " + str(time_dtree_pred) + " seconds")
time_dataInst_pred = time_dtree_pred/y_test.count()
print("Time to predict with decision tree model per data instance: " + str(time_dataInst_pred) + " seconds")



print(classification_report(y_test, predict_tree))



for x in range(0, 6):
    print("SET" + str(x+1))
    random_num = random.randint(1,101)
    print("Random split: " + str(random_num))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = random_num)
    print("Y Test Value Counts: ")
    print(str(y_test.value_counts()))
    dtree.fit(X_train, y_train)
    predict_tree_train = dtree.predict(X_train)
    print("Training set " + str(x+1))
    print(classification_report(y_train, predict_tree_train))
    predict_tree_test = dtree.predict(X_test)
    print("Testing set " + str(x+1))
    print(classification_report(y_test, predict_tree_test))
    print("------------------------------------------------")



print("Features sorted by their impurity score:")
print(sorted(zip(map(lambda x: round(x, 4), dtree.feature_importances_), features), 
             reverse=True))



from sklearn.feature_selection import RFE
rfe = RFE(dtree, 5)
rfe = rfe.fit(X_train, y_train)



# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)



dict_rfe = {'Feature name': features, 'RFE Ranking': rfe.ranking_}
dtree_rfe = pd.DataFrame(dict_rfe)
dtree_rfe.sort_values(by=['RFE Ranking'], inplace = True)
dtree_rfe


# # Random forest


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 150)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)



start = time.time()
rfc.fit(X_train, y_train)
end = time.time()
time_rfc_fit = end - start
print("Time to create random forest model: " + str(time_rfc_fit) + " seconds")



y_test.value_counts()



start = time.time()
predict_forest = rfc.predict(X_test)
end = time.time()
time_rfc_pred = end - start
print("Time to predict with random forest model: " + str(time_rfc_pred) + " seconds")
time_dataInst_rfc_pred = time_rfc_pred/y_test.count()
print("Time to predict with random forest model per data instance: " + str(time_dataInst_rfc_pred) + " seconds")



print(classification_report(y_test,predict_forest))



for x in range(0, 6):
    print("SET" + str(x+1))
    random_num = random.randint(1,101)
    print("Random split: " + str(random_num))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = random_num)
    print("Y Test Value Counts: ")
    print(str(y_test.value_counts()))
    rfc.fit(X_train, y_train)
    predict_forest_train = rfc.predict(X_train)
    print("Training set " + str(x+1))
    print(classification_report(y_train, predict_forest_train))
    predict_forest_test = rfc.predict(X_test)
    print("Testing set " + str(x+1))
    print(classification_report(y_test, predict_forest_test))
    print("------------------------------------------------")



print("Features sorted by their impurity score:")
print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), features), 
             reverse=True))



from sklearn.feature_selection import RFE
rfe2 = RFE(rfc, 5)
rfe2 = rfe2.fit(X_train, y_train)



print(rfe2.support_)
print(rfe2.ranking_)



dict_rfe2 = {'Feature name': features, 'RFE Ranking': rfe2.ranking_}
forest_rfe = pd.DataFrame(dict_rfe2)
forest_rfe.sort_values(by=['RFE Ranking'], inplace = True)
forest_rfe


# # Logistic regression


from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression()



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)



start = time.time()
logmodel.fit(X_train, y_train)
end = time.time()
time_log_fit = end - start
print("Time to create logistic regression model: " + str(time_log_fit) + " seconds")



y_test.value_counts()



start = time.time()
predict_log = logmodel.predict(X_test)
end = time.time()
time_log_pred = end - start
print("Time to predict with logistic regression model: " + str(time_log_pred) + " seconds")
time_dataInst_log_pred = time_log_pred/y_test.count()
print("Time to predict with logistic regression model per data instance: " + str(time_dataInst_log_pred) + " seconds")



print(classification_report(y_test, predict_log))



for x in range(0, 6):
    print("SET" + str(x+1))
    random_num = random.randint(1,101)
    print("Random split: " + str(random_num))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = random_num)
    print("Y Test Value Counts: ")
    print(str(y_test.value_counts()))
    logmodel.fit(X_train, y_train)
    predict_log_train = logmodel.predict(X_train)
    print("Training set " + str(x+1))
    print(classification_report(y_train, predict_log_train))
    predict_log_test = logmodel.predict(X_test)
    print("Testing set " + str(x+1))
    print(classification_report(y_test, predict_log_test))
    print("------------------------------------------------")



coef = []
for item in logmodel.coef_:
    coef.extend(item)
print(coef)



d = {'Features' : features, 'LogCoef': coef}
df_coef = pd.DataFrame(d)



df_coef['sort'] = df_coef.LogCoef.abs()



df_coef.sort_values(by = ['sort'], ascending = False, inplace = True)
df_coef.drop('sort', axis = 1, inplace = True)
df_coef



from sklearn.feature_selection import RFE
rfe3 = RFE(logmodel, 5)
rfe3 = rfe3.fit(X_train, y_train)



print(rfe3.support_)
print(rfe3.ranking_)



dict_rfe3 = {'Feature name': features, 'RFE Ranking': rfe3.ranking_}
logmodel_rfe = pd.DataFrame(dict_rfe3)
logmodel_rfe.sort_values(by=['RFE Ranking'], inplace = True)
logmodel_rfe


# # Storing and loading models


from sklearn.externals import joblib



dtree_file = 'dtree_model_recurrence.sav'
joblib.dump(dtree, dtree_file)



loaded_model = joblib.load(dtree_file)
result = loaded_model.predict(X_test)
print(result)



rforest_file = 'randomforest_model_recurrence.sav'
joblib.dump(rfc, rforest_file)



loaded_model = joblib.load(rforest_file)
result = loaded_model.predict(X_test)
print(result)



logreg_file = 'logreg_model_recurrence.sav'
joblib.dump(logmodel, logreg_file)



loaded_model = joblib.load(logreg_file)
result = loaded_model.predict(X_test)
print(result)


# # Visualizations


opioid.to_csv('recurG.csv')
# bars = ("8 or less", "9-11", "12", "13-15", "16 or more")
# y_pos = np.arange(len(bars))
# plt.xticks(y_pos, bars, fontweight='bold', fontsize='17', horizontalalignment='right')

plt.figure(figsize = (20,10))
sns.barplot(x = 'EDUC', y = 'NOPRIOR', data = opioid)



educ = opioid[['EDUC','NOPRIOR']]
educ = educ[(educ['EDUC'] != -9)]
educ.EDUC[educ.EDUC == 1] = '8 or less'
educ.EDUC[educ.EDUC == 2] = '9-11'
educ.EDUC[educ.EDUC == 3] = '12'
educ.EDUC[educ.EDUC == 4] = '13-15'
educ.EDUC[educ.EDUC == 5] = '16 or more'
plt.figure(figsize = (13,10))

sns.set_context("notebook", font_scale=3.0, rc={"lines.linewidth": 2.5})

ax = sns.barplot(x = 'EDUC', y = 'NOPRIOR', order=['8 or less', "9-11", '12', '13-15', '16 or more'], data = educ)

ax.set(xlabel='Education (highest school grade completed)', ylabel='Substance Abuse Recurrence Likelihood')

# Set these based on your column counts
columncounts = [30,30,30,30,30]

# Maximum bar width is 1. Normalise counts to be in the interval 0-1. Need to supply a maximum possible count here as maxwidth
def normaliseCounts(widths,maxwidth):
    widths = np.array(widths)/float(maxwidth)
    return widths

widthbars = normaliseCounts(columncounts,50)
plt.ylim(0,1)

# Loop over the bars, and adjust the width (and position, to keep the bar centred)
for bar,newwidth in zip(ax.patches,widthbars):
    x = bar.get_x()
    width = bar.get_width()
    centre = x+width/2.

    bar.set_x(centre-newwidth/2.)
    bar.set_width(newwidth)



fig = ax.get_figure()
fig.savefig("educ_noprior.png")



plt.figure(figsize = (20,10))
sns.barplot(x = 'MARSTAT', y = 'NOPRIOR', data = opioid)



marstat = opioid[['MARSTAT','NOPRIOR']]
marstat = marstat[(marstat['MARSTAT'] != -9)]
marstat.MARSTAT[marstat.MARSTAT == 1] = 'Unmarried'
marstat.MARSTAT[marstat.MARSTAT == 2] = 'Married'
marstat.MARSTAT[marstat.MARSTAT == 3] = 'Separated'
marstat.MARSTAT[marstat.MARSTAT == 4] = 'Divorced/\nwidowed'
plt.figure(figsize = (15,12))

sns.set_context("notebook", font_scale=2.8, rc={"lines.linewidth": 2.5})

ax = sns.barplot(x = 'MARSTAT', y = 'NOPRIOR', order=['Unmarried', "Married", 'Separated', 'Divorced/\nwidowed'], data = marstat)

ax.set(xlabel='Marital Status', ylabel='Substance Abuse Recurrence Likelihood')

# Set these based on your column counts
columncounts = [23,23,23,23]

# Maximum bar width is 1. Normalise counts to be in the interval 0-1. Need to supply a maximum possible count here as maxwidth
def normaliseCounts(widths,maxwidth):
    widths = np.array(widths)/float(maxwidth)
    return widths

widthbars = normaliseCounts(columncounts,50)
plt.ylim(0,1)

# Loop over the bars, and adjust the width (and position, to keep the bar centred)
for bar,newwidth in zip(ax.patches,widthbars):
    x = bar.get_x()
    width = bar.get_width()
    centre = x+width/2.

    bar.set_x(centre-newwidth/2.)
    bar.set_width(newwidth)



fig = ax.get_figure()
fig.savefig("marstat_noprior2.png")



plt.figure(figsize = (20,10))
sns.barplot(x = 'EMPLOY', y = 'NOPRIOR', data = opioid)



empl = opioid[['EMPLOY','NOPRIOR']]
empl = empl[(empl['EMPLOY'] != -9)]
empl.EMPLOY[empl.EMPLOY == 1] = 'Full time'
empl.EMPLOY[empl.EMPLOY == 2] = 'Part time'
empl.EMPLOY[empl.EMPLOY == 3] = 'Unemployed'
empl.EMPLOY[empl.EMPLOY == 4] = 'Not in\nlabor force'
plt.figure(figsize = (16,12))

sns.set_context("notebook", font_scale=2.8, rc={"lines.linewidth": 2.5})

ax = sns.barplot(x = 'EMPLOY', y = 'NOPRIOR', order=['Full time', "Part time", 'Unemployed', 'Not in\nlabor force'], data = empl)

ax.set(xlabel = 'Employment Status', ylabel='Substance Abuse Recurrence Likelihood')

# Set these based on your column counts
columncounts = [23,23,23,23]

# Maximum bar width is 1. Normalise counts to be in the interval 0-1. Need to supply a maximum possible count here as maxwidth
def normaliseCounts(widths,maxwidth):
    widths = np.array(widths)/float(maxwidth)
    return widths

widthbars = normaliseCounts(columncounts,50)
plt.ylim(0,1)

# Loop over the bars, and adjust the width (and position, to keep the bar centred)
for bar,newwidth in zip(ax.patches,widthbars):
    x = bar.get_x()
    width = bar.get_width()
    centre = x+width/2.

    bar.set_x(centre-newwidth/2.)
    bar.set_width(newwidth)



fig = ax.get_figure()
fig.savefig("employ_noprior2.png")



plt.figure(figsize = (20,10))
sns.barplot(x = 'PRIMINC', y = 'NOPRIOR', data = opioid)



prim = opioid[['PRIMINC','NOPRIOR']]
prim = prim[(prim['PRIMINC'] != -9)]
prim.PRIMINC[prim.PRIMINC == 1] = 'Wages/salary'
prim.PRIMINC[prim.PRIMINC == 2] = 'Public assistance'
prim.PRIMINC[prim.PRIMINC == 3] = 'Retirement/\npension'
prim.PRIMINC[prim.PRIMINC == 20] = 'Other'
prim.PRIMINC[prim.PRIMINC == 21] = 'None'
plt.figure(figsize = (28,14))

sns.set_context("notebook", font_scale=3.2, rc={"lines.linewidth": 2.5})

ax = sns.barplot(x = 'PRIMINC', y = 'NOPRIOR', order=['Wages/salary', "Public assistance", 'Retirement/\npension', 'Other', 'None'], data = prim)

ax.set(xlabel='Source of Income/Support', ylabel='Substance Abuse Recurrence Likelihood')

# Set these based on your column counts
columncounts = [23,23,23,23,23]

# Maximum bar width is 1. Normalise counts to be in the interval 0-1. Need to supply a maximum possible count here as maxwidth
def normaliseCounts(widths,maxwidth):
    widths = np.array(widths)/float(maxwidth)
    return widths

widthbars = normaliseCounts(columncounts,50)
plt.ylim(0,1)

# Loop over the bars, and adjust the width (and position, to keep the bar centred)
for bar,newwidth in zip(ax.patches,widthbars):
    x = bar.get_x()
    width = bar.get_width()
    centre = x+width/2.

    bar.set_x(centre-newwidth/2.)
    bar.set_width(newwidth)



fig = ax.get_figure()
fig.savefig("prim_noprior2.png")



plt.figure(figsize = (20,10))
sns.barplot(x = 'RACE', y = 'NOPRIOR', data = opioid)



plt.figure(figsize = (20,10))
sns.barplot(x = 'PMSA', y = 'NOPRIOR', data = opioid)



plt.figure(figsize = (20,10))
sns.barplot(x = 'STFIPS', y = 'NOPRIOR', data = opioid)





