#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %%
sim_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/sim.csv"
sim_df = pd.read_csv(sim_url)
sim_df
#%%
sns.set_theme(style="white")
mask = np.triu(np.ones_like(sim_df, dtype=bool))
_, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Plot a similarity heat map
sns.heatmap(sim_df, mask=mask, cmap=cmap, vmin=0.01, vmax=1, center=0,
            square=True)
#%%
course_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"
course_df = pd.read_csv(course_url)
bow_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/courses_bows.csv"
bow_df = pd.read_csv(bow_url)
bow_df.head()

#%%
def get_doc_dicts(bow_df):
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict
#%%
course1 = course_df[course_df['COURSE_ID'] == "ML0151EN"]
course1
course2 = course_df[course_df['COURSE_ID'] == "ML0101ENv3"]
course2
#%%
idx_id_dict, id_idx_dict = get_doc_dicts(bow_df)
idx1 = id_idx_dict["ML0151EN"]
idx2 = id_idx_dict["ML0101ENv3"]
print(f"Course 1's index is {idx1} and Course 2's index is {idx2}")
#%%
sim_matrix = sim_df.to_numpy()
sim = sim_matrix[idx1][idx2]
sim
#%%

#%%
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', None)
course_df[['COURSE_ID', 'TITLE']]
#%%
pd.reset_option('display.max_rows')
pd.reset_option('max_colwidth')
#%%
enrolled_course_ids = ["excourse93","GPXX05RDEN","excourse92","excourse68","excourse69"] # add your interested coures id to the list
enrolled_courses = course_df[course_df['COURSE_ID'].isin(enrolled_course_ids)]
enrolled_courses
#%%
all_courses = set(course_df['COURSE_ID'])
unselected_course_ids = all_courses.difference(enrolled_course_ids)
unselected_course_ids
#%%
def generate_recommendations_for_one_user(enrolled_course_ids, unselected_course_ids, id_idx_dict, sim_matrix):
    # Create a dictionary to store your recommendation results
    res = {}
    threshold = 0.6 
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                
                sim = 0
                # Find the two indices for each enrolled_course and unselect_course, based on their two ids
                # Calculate the similarity between an enrolled_course and an unselect_course
                # e.g., Course ML0151EN's index is 200 and Course ML0101ENv3's index is 158
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]

                # Find the similarity value from the sim_matrix
                # sim = sim_matrix[200][158]
                sim = sim_matrix[idx1][idx2]

                if sim > threshold:
                    if unselect_course not in res:
                        res[unselect_course] = sim
                    else:
                        if sim >= res[unselect_course]:
                            res[unselect_course] = sim
                            
    # Sort the results by similarity
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res

generate_recommendations_for_one_user(enrolled_course_ids,unselected_course_ids,id_idx_dict,sim_matrix)
#%%
test_users_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/rs_content_test.csv"
test_users_df = pd.read_csv(test_users_url)
test_users = test_users_df.groupby(['user']).max().reset_index(drop=False)
test_user_ids = test_users['user'].to_list()
print(f"Total numbers of test users {len(test_user_ids)}")
# %%
def generate_recommendations_for_all():
    users = []
    courses = []
    sim_scores = []
    # Test user dataframe
    test_users_df = pd.read_csv(test_users_url)

    # Course similarity matrix
    sim_df = pd.read_csv(sim_url)
    sim_matrix = sim_df.to_numpy()

    # Course content dataframe
    course_df = pd.read_csv(course_url)

    # Course BoW features
    bow_df = pd.read_csv(bow_url)
    test_users = test_users_df.groupby(['user']).max().reset_index(drop=False)
    test_user_ids = test_users['user'].to_list()
    idx_id_dict, id_idx_dict = get_doc_dicts(bow_df)

    # ...
    all_courses = set(course_df['COURSE_ID'])

    for user_id in test_user_ids:
        users.append(user_id)
        # For each user, call generate_recommendations_for_one_user() to generate the recommendation results
        # Save the result to courses, sim_scores list
        enrolled_course_ids = test_users_df[test_users_df.user==user_id].item.tolist() #Extract the enrolled courses by the user
        unselected_course_ids = all_courses.difference(enrolled_course_ids) #unselected_course_ids
        outcome = generate_recommendations_for_one_user(enrolled_course_ids, unselected_course_ids, id_idx_dict, sim_matrix) # Function from above
        courses.append(list(outcome.keys())) # courses names
        sim_scores.append(list(outcome.values())) # courses similarity index
        pass
    
    return users, courses, sim_scores
# %%
res_dict = {}
users, courses, sim_scores = generate_recommendations_for_all()
res_dict['USER'] = users
res_dict['COURSE_ID'] = courses
res_dict['SCORE'] = sim_scores
res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
# %%
s = 0
for i in range(len(res_df['COURSE_ID'])):
    s+=len(res_df['COURSE_ID'].iloc[i])
avg = s/len(res_df['COURSE_ID'])
avg
# %%
course_union = set()
for i in range(len(res_df['COURSE_ID'])):
    course_union = course_union.union(set(res_df['COURSE_ID'].iloc[i]))

recc = list(course_union)
tally = [0]*len(recc)

for i in range(len(res_df['COURSE_ID'])):
    for j in res_df['COURSE_ID'].iloc[i]:
        for idx,k in enumerate(recc):
            if j==k:
                tally[idx]+=1
                
pd.Series({k: v for k, v in sorted(dict(zip(recc,tally)).items(), key=lambda item: item[1])}).sort_values(ascending=False)[:10]
