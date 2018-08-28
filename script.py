#Script to pre-process the dataset and train prediction models/classifiers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression




#taken from https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows
def tidy_split(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df	

def isForeign(x):
	try:
		x.encode('ascii')
	except UnicodeEncodeError:
		return True
	else:
		return False

def winRate(word):
	#ids=chats.loc[chats['key'] == word].chat_id.value_counts()
	ids=chats.loc[chats['key'] == word].chat_id.drop_duplicates()
	ids = pd.DataFrame({'chat_id':ids.values})
	a=pd.merge(wins,ids, left_index=True, right_on='chat_id')
	return round((a.won.value_counts(normalize=True)[1])*100,2)

global chats, wins
#Read datasets
chats=pd.read_csv('chat.csv')
chats=chats.drop('unit',1)	#get rid of player name; already have it coded as a slot
wins=pd.read_csv('match.csv')
wins=wins[['match_id','radiant_win']]	#only keep information if team A won or not

chats['chat_id']=0	#introduce chat id - idea is to separate team A chat from team B chat
chats['chat_id']=chats['match_id']*2
chats['chat_id'][chats.slot>4]+=1	#Players 0-4 are in team 1, 5-9 are in team 2
chats=chats.sort_values(['chat_id','time'], ascending=[True,True])

#Pre-processing to 
wins=wins.merge(chats[['chat_id','match_id']], on='match_id')
wins=wins.drop_duplicates(keep='first')
wins[(wins.chat_id%2==0) & (wins.radiant_win)]
wins['won']=0
wins['won'][(wins.chat_id%2==0) & (wins.radiant_win)]=1
wins['won'][(wins.chat_id%2==1) & ~(wins.radiant_win)]=1	#now every chat id is matched with won/didn't entry
wins=wins.drop(['radiant_win', 'match_id'], axis=1)

chats=tidy_split(chats, 'key', sep=' ')				#split chat row entries on empty space to keep them consistent
wins.sort_values('chat_id', ascending=True)
wins = wins.reset_index(drop=True)
chats = chats.reset_index(drop=True)
chats=chats.sort_values(['chat_id','time'], ascending=[True,True])
chats=chats.drop(chats[chats.key.str.strip()==''].index)	#drop whitespace since last step introduced a few empty rows

#Find if non-english characters are present in an entry
#wins['foreign']=0
#wins=wins.set_index('chat_id')
#wins['foreign'].loc[chats.loc[chats.key.apply(isForeign)].chat_id.drop_duplicates()]=1 #set indexes of foreign chats


wins=wins.set_index('chat_id')
temp=chats.chat_id.value_counts(sort=False)
temp = pd.DataFrame({'chat_id':temp.index, 'chat_length':temp.values})
wins=pd.merge(wins,temp, left_index=True, right_on='chat_id')
wins=wins.set_index('chat_id')

#get words that occurr at least once in more than 10% of the chats
uniqueWords=chats.sort_values('chat_id').drop_duplicates(subset=['chat_id', 'key'], keep='last')	#remove duplicate words for same chat_id
uniqueWords=uniqueWords[['key', 'chat_id']]
uniqueWords['val']=1

counts=uniqueWords.key.value_counts()
uniqueWords = uniqueWords.loc[uniqueWords['key'].isin(counts[counts > 10000].index), :]

temp=uniqueWords.pivot(index='chat_id', columns='key',values='val')
wins=wins.merge(temp, left_index=True, right_index=True).fillna(0)

for x in list(wins.columns)[2:]:
	if winRate(x)<55 and winRate(x) >45:
		wins=wins.drop([x],axis=1)
'''#to plot the used words
counts=chats.loc[chats['key'].isin(list(wins.columns)[2:])].key.value_counts()
plt.bar(counts.index.tolist(),counts)
plt.xlabel('Word')	
plt.ylabel('Number of Occurances')


#to plot chat length to wins
plt.hist(wins.loc[wins.won == 0].chat_length, 30, range=[0, 350], facecolor='red', alpha=0.5, ec = 'black')
plt.hist(wins.loc[wins.won == 1].chat_length, 30, range=[0, 350], facecolor='green', alpha=0.5, ec = 'black')
plt.xlabel('Chat Length')	
plt.legend(['Lost Games','Won Games'])
plt.ylabel('Number of Chats')
plt.show()

#to plot amount of games with foreign phrases
plt.pie(wins.foreign.value_counts(),autopct='%1.1f%%',labels=['Latin-Only','Contain Foreign Characters'])
plt.axis('equal')
plt.title('Amount of Chatlogs with Foreign Characters')
plt.show()

#to plot win rates with foreign
plt.pie(wins.loc[wins.foreign==True].won.value_counts(),autopct='%1.1f%%',labels=['Won','Lost'])
plt.axis('equal')
plt.title('Win Rate with Foreign Characters')
plt.show()

#to plot win rates without foreign
plt.pie(wins.loc[wins.foreign==False].won.value_counts(),autopct='%1.1f%%',labels=['Won','Lost'])
plt.axis('equal')
plt.title('Win Rate without Foreign Characters')
plt.show()

#to plot win rate per word
tempList=list(wins.columns)[2:]
wordWinRates=[]
for x in tempList:
	wordWinRates.append(winRate(x))
tempList[tempList.index('?')]='\'?\''
plt.rcParams.update({'font.size':10})
plt.barh(tempList, wordWinRates,align='center')
plt.gca().invert_yaxis()
plt.xlabel('Win Percentage')
plt.ylabel('Word')
plt.show()
'''


'''
#unsupervised learning
#https://www.datacamp.com/community/tutorials/machine-learning-python
data=wins.drop(['won','chat_length'],axis=1)
target=wins.won
data = scale(data)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
clf = cluster.KMeans(init='k-means++', n_clusters=2)
clf.fit(X_train)
y_pred=clf.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
X_pca = PCA(n_components=2).fit_transform(X_train)
# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)
# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
# Add scatterplots to the subplots 
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')
# Show the plots
plt.show()
'''



data=wins.drop(['won'],axis=1)
target=wins.won
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

'''
#Logistic Regression - 63%
lm = LogisticRegression()
lm.fit(X_train, y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))

data = scale(data)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
X_pca = PCA(n_components=2).fit_transform(X_train)
# Compute cluster centers and predict cluster index for each sample
clusters = lm.predict(X_train)
# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
# Add scatterplots to the subplots 
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')
# Show the plots
plt.show()




#Neural Network - 63%'''
mlp = MLPClassifier(hidden_layer_sizes=(14,14,14))
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
'''

x_test=scale(x_test)
y_test=scale(y_test)
#X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
X_pca = PCA(n_components=2).fit_transform(X_test)
# Compute cluster centers and predict cluster index for each sample
clusters = mlp.predict(X_test)
# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# Adjust layout
fig.suptitle('Predicted Versus Testing Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
# Add scatterplots to the subplots 
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
ax[0].set_title('Predicted Testing Labels')
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_test)
ax[1].set_title('Actual Testing Labels')
# Show the plots
plt.show()

#Random Forest - 58%
forest=RandomForestClassifier()
regr = forest.fit(X_train, y_train)
predictions = regr.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
'''
