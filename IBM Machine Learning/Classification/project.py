#%%
import pandas as pd
import numpy as np 

from matplotlib.pyplot import figure
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn import metrics
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from collections import Counter
#%%
df = pd.read_csv('train.csv')

#%%
print(df.info())

# %%
print(df.head())
# %%
df.columns
# %%
import matplotlib.pyplot as plt

fig,axes = plt.subplots(5,5, figsize=(25,15))

cols = df.columns
index = 0
for i in range(5):
  for j in range(5):
    sns.boxplot(y=cols[index], data=df, ax=axes[i,j])
    index += 1
# %%
X = df.drop(columns=['price_range'])
y = df['price_range'].copy()
# %%
y.value_counts().plot.bar(color=['green', 'red'])
# %%
#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 42)
y_test.value_counts().plot.bar(color=['green', 'red'])
# %%
y_test.value_counts()
# %%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)
#%%
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = score(yt, yp)
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos
evaluate_metrics(y_test, preds_lr)


# %%
import seaborn as sns
cf = confusion_matrix(y_test, preds_lr)
plt.figure(figsize=(16, 12))
ax = sns.heatmap(cf, annot=True, fmt="d", xticklabels=["0", "1", "2", "3"], 
                 yticklabels=["0", "1", "2", "3"])
ax.set(title="Confusion Matrix")
# %%
#knn
from sklearn.neighbors import KNeighborsClassifier
max_k = 50
# Create an empty list to store f1score for each k
f1_scores = []
for k in range(1, max_k + 1):
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors = k)
    # Train the classifier
    knn = knn.fit(X_train, y_train)
    preds_knn = knn.predict(X_test)
    # Evaluate the classifier with f1score
    f1 = f1_score(y_test, preds_knn, average='weighted')
    f1_scores.append((k, round(f1, 4)))
# Convert the f1score list to a dataframe
f1_results = pd.DataFrame(f1_scores, columns=['K', 'F1 Score'])
f1_results.set_index('K')
ax = f1_results.plot(figsize=(12, 12))
ax.set(xlabel='Num of Neighbors', ylabel='F1 Score')
ax.set_xticks(range(1, max_k, 2))
plt.ylim((0.85, 1))
plt.title('KNN F1 Score')

# %%
knn_opt = KNeighborsClassifier(n_neighbors = 9)
knn_opt = knn_opt.fit(X_train, y_train)
preds_opt_knn = knn_opt.predict(X_test)
evaluate_metrics(y_test, preds_opt_knn)

# %%
cf = confusion_matrix(y_test, preds_opt_knn)
plt.figure(figsize=(16, 12))
ax = sns.heatmap(cf, annot=True, fmt="d", xticklabels=["0", "1", "2", "3"], 
                 yticklabels=["0", "1", "2", "3"])
ax.set(title="Confusion Matrix")

#%%
#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
param_grid = {'n_estimators': [2*n+1 for n in range(20)],
             'max_depth' : [2*n+1 for n in range(10) ],
             'max_features':["auto", "sqrt", "log2"]}
search = GridSearchCV(estimator=model, param_grid=param_grid,scoring='accuracy')
search.fit(X_train, y_train)

# %%
print(search.best_params_)
# %%
rf = RandomForestClassifier( max_depth = 19, max_features = 'auto', n_estimators = 33)
rf.fit(X_train,y_train)
preds_rf = rf.predict(X_test)
evaluate_metrics(y_test, preds_rf)

# %%
cf = confusion_matrix(y_test, preds_rf)
plt.figure(figsize=(16, 12))
ax = sns.heatmap(cf, annot=True, fmt="d", xticklabels=["0", "1", "2", "3"], 
                 yticklabels=["0", "1", "2", "3"])
ax.set(title="Confusion Matrix")
# %%
