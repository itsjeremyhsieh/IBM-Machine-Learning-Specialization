#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#%%
rating_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
user_emb_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/user_embeddings.csv"
item_emb_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_embeddings.csv"
#%%
rating_df = pd.read_csv(rating_url)
rating_df.head()

#%%
# Load user embeddings
user_emb = pd.read_csv(user_emb_url)
# Load item embeddings
item_emb = pd.read_csv(item_emb_url)
#%%
# Merge user embedding features
user_emb_merged = pd.merge(rating_df, user_emb, how='left', left_on='user', right_on='user').fillna(0)
# Merge course embedding features
merged_df = pd.merge(user_emb_merged, item_emb, how='left', left_on='item', right_on='item').fillna(0)
merged_df.head()

# %%
u_feautres = [f"UFeature{i}" for i in range(16)]
c_features = [f"CFeature{i}" for i in range(16)]

user_embeddings = merged_df[u_feautres]
course_embeddings = merged_df[c_features]
ratings = merged_df['rating']

# Aggregate the two feature columns using element-wise add
regression_dataset = user_embeddings + course_embeddings.values
regression_dataset.columns = [f"Feature{i}" for i in range(16)]
regression_dataset['rating'] = ratings
regression_dataset.head()
#%%
X = regression_dataset.iloc[:, :-1]
y = regression_dataset.iloc[:, -1]
print(f"Input data shape: {X.shape}, Output data shape: {y.shape}")
# %%
y.unique()
# %%
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
lr = LinearRegression()
lr.fit(x_train,y_train)

# %%
predictions = lr.predict(x_test)
rmse = np.sqrt(mean_squared_error(predictions,y_test))
print('LR RMSE',rmse)
# %%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso,Ridge,ElasticNet
#LASSO
las = Lasso()
params = {'alpha':np.geomspace(0.1,1,20)}
grid_las = GridSearchCV(estimator = las, param_grid = params, cv = 3)
grid_las.fit(x_train,y_train)
pred_las = grid_las.predict(x_test)
print('Lasso RMSE',np.sqrt(mean_squared_error(pred_las,y_test)))
# %%
r = Ridge()
params = {'alpha':np.geomspace(0.1,1,20)}
grid_r = GridSearchCV(estimator = r, param_grid = params, cv = 3)
grid_r.fit(x_train,y_train)
pred_r = grid_r.predict(x_test)
print('Ridge RMSE',np.sqrt(mean_squared_error(pred_r,y_test)))
#%%
elas = ElasticNet()
params = {'alpha':np.geomspace(0.1,1,5),
        'l1_ratio':np.geomspace(0.1,1,5)}
grid_elas = GridSearchCV(estimator = elas, param_grid = params, cv = 3)
grid_elas.fit(x_train,y_train)
pred_elas = grid_elas.predict(x_test)
print('Elastic Net RMSE',np.sqrt(mean_squared_error(pred_elas,y_test)))
# %%
