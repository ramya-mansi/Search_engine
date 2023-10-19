import pandas as pd
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from gensim.models import LdaModel
import re

# Define a function to preprocess text
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabet characters
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]

    unigrams = words
    bigrams = [' '.join(phrase) for phrase in list(ngrams(words, 2))]
    
    return unigrams, bigrams  # Return both unigrams and bigrams

# Load your CSV file into a DataFrame
df = pd.read_csv('thurs_new.csv')

# Apply preprocessing to your data
df['unigrams'], df['bigrams'] = zip(*df['Cleaned Data'].apply(preprocess_text))

# Create a dictionary for unigrams
dictionary_unigrams = corpora.Dictionary(df['unigrams'])
corpus_unigrams = [dictionary_unigrams.doc2bow(text) for text in df['unigrams']]

# Train the LDA model for unigrams
lda_model_unigrams = LdaModel(corpus_unigrams, num_topics=10, id2word=dictionary_unigrams, passes=15)

# Create a dictionary for bigrams
dictionary_bigrams = corpora.Dictionary(df['bigrams'])
corpus_bigrams = [dictionary_bigrams.doc2bow(text) for text in df['bigrams']]

# Train the LDA model for bigrams
lda_model_bigrams = LdaModel(corpus_bigrams, num_topics=10, id2word=dictionary_bigrams, passes=15)

# Define a function to get the most significant words for a topic
def get_topic_words(model, dictionary, topic):
    topic_words = [word for word, prob in model.show_topic(topic, topn=10) if word not in ['agents','agent','ai', 'intelligence','artificial intelligence','cial intelligence','artificial','arti cial','phys rev','phys',]]  # Adjust topn as needed
    return topic_words

# Extract 3 unigram topics and 2 bigram topics
df['unigram_topics'] = df['unigrams'].apply(lambda x: get_topic_words(lda_model_unigrams, dictionary_unigrams, lda_model_unigrams[dictionary_unigrams.doc2bow(x)][0][0]))
df['bigram_topics'] = df['bigrams'].apply(lambda x: get_topic_words(lda_model_bigrams, dictionary_bigrams, lda_model_bigrams[dictionary_bigrams.doc2bow(x)][0][0]))

# Combine the topics into a single column
df['topics'] = df.apply(lambda row: row['unigram_topics'][:3] + row['bigram_topics'][:2], axis=1)

# Save the DataFrame to a new CSV file with the 'topics' column
df.to_csv('output_with_topics_seven.csv', index=False)

