# HHA1491 - Henry Harvin Sentimental Analysis
# Dataset: Google Reviews

# Importing Libraries
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Loading Dataset: File name - Trust Pilot.xlsx
dataset = pd.read_csv('Google Reviews.csv')
dataset
dataset.drop(['Rating'], axis=1, inplace=True)
dataset.info()

# Converting Dataset into String Type 
dataset = dataset.to_string(index = False) 
dataset = dataset.lower()

#-------------------Cleaning the data-----------------------------------
import re
dataset = re.sub("[^A-Za-z0-9]+"," ",dataset)
# sub is used for substitution
# ^ is used for negation so it means accept these all ranges convert others to space
# + signify that atleast one should be converted

#----------------------Tokenization--------------------------------------------
# Splitting the complete Dataset/ all Comments into words/Tokens 
import nltk    
from nltk.tokenize import word_tokenize
Tokens = word_tokenize(dataset) # Create a Lists of all words

# No. of Tokens in the dataset
len(Tokens)

#-------------Remove the Stop Words---------------------
import nltk.corpus

# Enlisting the stopwords present in English language
stopwords = nltk.corpus.stopwords.words('english')
stopwords[0:10]

# Eliminating the stopwords from the Tokens
sentence = []   
for FinalWord in Tokens:
    if FinalWord not in stopwords:
        sentence.append(FinalWord)  
len(sentence) # Lists of words after removing the stopwords

# Removing Irrelevant words
sentence = [token for token in sentence if token not in ['I','The','n','course','henry','harvin']]

# Freqency of occurence of distinct words in the List
from nltk.probability import FreqDist
fdist = FreqDist()
        
for word in sentence:
    fdist[word] += 1 
    
fdist.plot(20, title='Top 20 Most Occoured Words in the "Google Reviews Dataset"') # Plotting the Frequency Data


# Joining all the Tokens together to create a long sinlge string for Analysis
filtered_sentence = " "
filtered_sentence = filtered_sentence.join(sentence)

#-------------------------Stemming----------------------------------------
# The Process of resetting each words to their Base form like Having --> Have
from nltk.stem import PorterStemmer
pst=PorterStemmer()
filtered_sentence = pst.stem(filtered_sentence)

# Calculating final Sentiment Score
Score = TextBlob(filtered_sentence)
print(Score.sentiment)

# Creating the WordCloud
from wordcloud import WordCloud
word_cloud = WordCloud(width = 512, height = 512, background_color='white', stopwords=stopwords).generate(filtered_sentence)
plt.imshow(word_cloud)
word_cloud.to_file("Google Reviews_WordCloud.png")


