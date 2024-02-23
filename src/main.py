import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import pyarrow
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer

#Some setups
nltk.download('vader_lexicon')
plt.style.use('ggplot')

# Read data
df = pd.read_csv('atomic-habits-reviews.csv')

ax = df['rating'].value_counts().sort_index().plot(kind='bar', title='Ratings', figsize=(10, 5))
ax.set_xlabel('Rating')
# plt.show()

sia = SentimentIntensityAnalyzer()

# print(sia.polarity_scores('I am so happy!'))
# Run the polarity score through the dataset
res={}
for i, d in tqdm(df.iterrows(), total=len(df)):
    text = d['text']
    reviewId = d['id']
    res[reviewId] = sia.polarity_scores(text)
# Merge the data to the new dataset
v = pd.DataFrame(res).T
v = v.reset_index().rename(columns={'index': 'id'})
v = v.merge(df, how='left', on='id')

# sPlot = sns.barplot(data=v, x='rating', y='compound')
# sPlot.set_title("Compound vs Rating")
# plt.show()


fig, axs = plt.subplots(1, 3, figsize=(30, 5))
sns.barplot(data=v, x='rating', y='pos', ax=axs[0])
sns.barplot(data=v, x='rating', y='neu', ax=axs[1])
sns.barplot(data=v, x='rating', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.show()