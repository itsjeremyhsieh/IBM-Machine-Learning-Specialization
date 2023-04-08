#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# %%
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
course_df = pd.read_csv(course_genre_url)
ratings_df = pd.read_csv(ratings_url)
course_df.columns
course_df.head()
# %%
titles = " ".join(title for title in course_df['TITLE'].astype(str))
# English Stopwords
stopwords = set(STOPWORDS)
stopwords.update(["getting started", "using", "enabling", "template", "university", "end", "introduction", "basic"])
wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400)
wordcloud.generate(titles)
plt.axis("off")
plt.figure(figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation='bilinear')
plt.show()
# %%
ml_courses = course_df[course_df['MachineLearning'] ==1]
ml_courses

# %%
ml_bd_courses = ml_courses[ml_courses['BigData']==1]
ml_bd_courses
# %%
genres = course_df.columns[2:]
genres
# %%
genre_counts = course_df.iloc[:,2:].sum().sort_values(ascending=False)
genre_counts
plot = genre_counts.plot.bar()
plot.set_xlabel("Genre")
plot.set_ylabel("Count")
# %%
#Analyze Course Enrollments
ratings_df.head()
# %%
ratings = ratings_df.groupby(['user']).size()
ratings.describe()
# %%
plot = sns.histplot(ratings,legend=False)
plot.set_xlabel("Enrolls")
plot.set_ylabel("Count")
# %%
# Find the Top-20 Most Popular Courses
top = ratings_df.groupby(['item']).size().sort_values(ascending=False).to_frame().iloc[:20,:]
top
merged_count = top.merge(course_df, left_on=['item'], right_on=['COURSE_ID']).loc[:,[0,'TITLE']]
merged_count = merged_count.reindex(columns=['TITLE',0])
merged_count
# %%
