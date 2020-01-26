## Decision tree, Random forest and Logistic regression models
## for predicting Cocaine addiction
## Important features are also extracted
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')



pre_opioid = pd.read_csv('Opioid.csv')



pre_opioid.head()



opioid = pre_opioid[['AGE','GENDER', 'RACE', 'ETHNIC', 'MARSTAT', 'EDUC', 'EMPLOY', 'VET', 'LIVARAG', 'PRIMINC', 'STFIPS', 'CBSA', 'PMSA', 'REGION', 'DIVISION', 'NOPRIOR', 'PSYPROB','HLTHINS', 'PRIMPAY', 'COKEFLG']].copy()



opioid.columns



opioid



features = list(opioid)
features.remove('COKEFLG')
features



opioid['COKEFLG'].value_counts()


# # Splitting data into training/testing sets


from sklearn.model_selection import train_test_split



X = opioid.drop('COKEFLG', axis = 1)
y = opioid['COKEFLG']



y



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random
import time


# # Decision tree


dtree = DecisionTreeClassifier()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



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



predict_tree



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



dtree_file = 'dtree_model_cocaine.sav'
joblib.dump(dtree, dtree_file)



loaded_model = joblib.load(dtree_file)
result = loaded_model.predict(X_test)
print(result)



rforest_file = 'randomforest_model_cocaine.sav'
joblib.dump(rfc, rforest_file)



loaded_model = joblib.load(rforest_file)
result = loaded_model.predict(X_test)
print(result)



logreg_file = 'logreg_model_cocaine.sav'
joblib.dump(logmodel, logreg_file)



loaded_model = joblib.load(logreg_file)
result = loaded_model.predict(X_test)
print(result)


# # Visualizations


plt.figure(figsize = (20,10))
sns.barplot(x = 'AGE', y = 'COKEFLG', data = opioid)



age = opioid[['AGE','COKEFLG']]
age = age[(age['AGE'] != -9)]
age.AGE[age.AGE == 2] = '12-14'
age.AGE[age.AGE == 3] = '15-17'
age.AGE[age.AGE == 4] = '18-20'
age.AGE[age.AGE == 5] = '21-24'
age.AGE[age.AGE == 6] = '25-29'
age.AGE[age.AGE == 7] = '30-34'
age.AGE[age.AGE == 8] = '35-39'
age.AGE[age.AGE == 9] = '40-44'
age.AGE[age.AGE == 10] = '45-49'
age.AGE[age.AGE == 11] = '50-54'
age.AGE[age.AGE == 12] = '>=55'
plt.figure(figsize = (20,10))

sns.set_context("notebook", font_scale=3.0, rc={"lines.linewidth": 2.5})

ax = sns.barplot(x = 'AGE', y = 'COKEFLG', order=['12-14', "15-17", '18-20', '21-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '>=55'], data =age)

ax.set(xlabel='Age (years)', ylabel='Cocaine Addiction Likelihood')

# Set these based on your column counts
columncounts = [30,30,30,30,30,30,30,30,30,30,30]

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
fig.savefig("age_coc.png")



plt.figure(figsize = (20,10))
sns.barplot(x = 'NOPRIOR', y = 'COKEFLG', data = opioid)



nopr = opioid[['NOPRIOR','COKEFLG']]
nopr = nopr[(nopr['NOPRIOR'] != -9)]

plt.figure(figsize = (15,10))

sns.set_context("notebook", font_scale=3.0, rc={"lines.linewidth": 2.5})

ax = sns.barplot(x = 'NOPRIOR', y = 'COKEFLG', data =nopr)

ax.set(xlabel='# of Prior Treatment Episodes', ylabel= 'Cocaine Addiction Likelihood')

# Set these based on your column counts
columncounts = [30,30,30,30,30,30]

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
fig.savefig("nopr_coc.png")



plt.figure(figsize = (20,10))
sns.barplot(x = 'EDUC', y = 'COKEFLG', data = opioid)



educ = opioid[['EDUC','COKEFLG']]
educ = educ[(educ['EDUC'] != -9)]
educ.EDUC[educ.EDUC == 1] = '8 or less'
educ.EDUC[educ.EDUC == 2] = '9-11'
educ.EDUC[educ.EDUC == 3] = '12'
educ.EDUC[educ.EDUC == 4] = '13-15'
educ.EDUC[educ.EDUC == 5] = '16 or more'
plt.figure(figsize = (13,10))

sns.set_context("notebook", font_scale=3.0, rc={"lines.linewidth": 2.5})

ax = sns.barplot(x = 'EDUC', y = 'COKEFLG', order=['8 or less', "9-11", '12', '13-15', '16 or more'], data = educ)

ax.set(xlabel='Education (highest school grade completed)', ylabel='Cocaine Addiction Likelihood')

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
fig.savefig("educ_coc.png")



marstat = opioid[['MARSTAT','COKEFLG']]
marstat = marstat[(marstat['MARSTAT'] != -9)]
marstat.MARSTAT[marstat.MARSTAT == 1] = 'Unmarried'
marstat.MARSTAT[marstat.MARSTAT == 2] = 'Married'
marstat.MARSTAT[marstat.MARSTAT == 3] = 'Separated'
marstat.MARSTAT[marstat.MARSTAT == 4] = 'Divorced/\nwidowed'
plt.figure(figsize = (15,14))

sns.set_context("notebook", font_scale=2.8, rc={"lines.linewidth": 2.5})

ax = sns.barplot(x = 'MARSTAT', y = 'COKEFLG', order=['Unmarried', "Married", 'Separated', 'Divorced/\nwidowed'], data = marstat)

ax.set(xlabel='Marital Status', ylabel='Cocaine Addiction Likelihood')

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
fig.savefig("marstat_coc2.png")



plt.figure(figsize = (20,10))
sns.barplot(x = 'MARSTAT', y = 'COKEFLG', data = opioid)



plt.figure(figsize = (20,10))
sns.barplot(x = 'PRIMINC', y = 'COKEFLG', data = opioid)



plt.figure(figsize = (20,10))
sns.barplot(x = 'EMPLOY', y = 'COKEFLG', data = opioid)



plt.figure(figsize = (20,10))
sns.barplot(x = 'RACE', y = 'COKEFLG', data = opioid)



race = opioid[['RACE','COKEFLG']]
race = race[(race['RACE'] != -9)]
race.RACE[race.RACE == 1] = 'Alaska\nNative'
race.RACE[race.RACE == 2] = 'American\nIndian'
race.RACE[race.RACE == 3] = 'Asian/\nPacific Islander'
race.RACE[race.RACE == 4] = 'Black'
race.RACE[race.RACE == 5] = 'White'
race.RACE[race.RACE == 13] = 'Asian'
race.RACE[race.RACE == 20] = 'Other race'
race.RACE[race.RACE == 21] = '2+ races'
race.RACE[race.RACE == 23] = 'Native Hawaiian/\nother Pacific Islander'

plt.figure(figsize = (30,10))

from textwrap import wrap

sns.set_context("notebook", font_scale=2.6, rc={"lines.linewidth": 2.5})

ax = sns.barplot(x = 'RACE', y = 'COKEFLG', order=['Alaska\nNative', "American\nIndian", 'Asian/\nPacific Islander','Black', 'White', 'Asian', 'Other race', '2+ races', 'Native Hawaiian/\nother Pacific Islander'], data = race)

ax.set(xlabel='Race', ylabel='Cocaine Addiction Likelihood')

# Set these based on your column counts
columncounts = [23,23,23,23,23,23,23,23,23]

# Maximum bar width is 1. Normalise counts to be in the interval 0-1. Need to supply a maximum possible count here as maxwidth
def normaliseCounts(widths,maxwidth):
    widths = np.array(widths)/float(maxwidth)
    return widths

widthbars = normaliseCounts(columncounts,50)
plt.ylim(0,1)
# plt.xticks(wrap = True)
# plt.xticks(rotation=45)

# ax.set_xticklabels([ '\n'.join(wrap(l, 20)) for l in ax.get_xticklabels])

# Loop over the bars, and adjust the width (and position, to keep the bar centred)
for bar,newwidth in zip(ax.patches,widthbars):
    x = bar.get_x()
    width = bar.get_width()
    centre = x+width/2.

    bar.set_x(centre-newwidth/2.)
    bar.set_width(newwidth)



fig = ax.get_figure()
fig.savefig("race_coc.png")





