import requests
from bs4 import BeautifulSoup
import re
import string
import nltk
import heapq
#import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer


SUMMARY_LENGTH = 6 #Number of sentences in summary
SENTENCE_LENGTH = 30 #Number of words per sentence
URL = 'https://www.nltk.org/' #Link to what you want to summarize


#scrape web page
def getText(url):
	page = requests.get(url).text
	soup = BeautifulSoup(page, 'html.parser')
	text = [p.text for p in soup.find_all('p')]
	#print(text)

	return text


def combineText(text):
	combined_text = ' '.join(text)

	return combined_text


#remove reference numbers and brackets
def cleanText_round1(text):
	text = re.sub(r'\[[0-9]*\]', ' ', text)
	text = re.sub(r'\s+', ' ', text)

	return text


def cleanText(text):
	text = text.lower()
	text = re.sub('\[.*?\]', '', text)
	text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
	text = re.sub('\w*\d\w*', '', text)
	text = re.sub('[''""...‘’“”…]', '', text)
	text = re.sub('\n', '', text)

	return text


#frequency of occurances in the text
def countVectorize(text):
	stop_words = nltk.corpus.stopwords.words('english')
	text_dtm = {}

	for word in nltk.word_tokenize(text):
		if word not in stop_words:
			if word not in text_dtm.keys():
				text_dtm[word] = 1
			else:
				text_dtm[word] += 1

	max_freq = max(text_dtm.values())
	for word in text_dtm.keys():
		text_dtm[word] = text_dtm[word]/max_freq

	#Word frequency with scikit-learn CountVectorizer
	'''
	cv = CountVectorizer(stop_words='english')
	text_cv = cv.fit_transform(text_clean)
	text_dtm = pd.DataFrame(text_cv.toarray(), columns=cv.get_feature_names())
	print(text_dtm)
	'''

	return text_dtm


#weighting for each sentence
def sentenceScore(sentences, text_dtm):
	sentence_score = {}
	for sentence in sentences:
		for word in nltk.word_tokenize(sentence.lower()):
			if word in text_dtm.keys():
				if len(sentence.split(' ')) < SENTENCE_LENGTH: #Only sentences shorter than 30 words
					if sentence not in sentence_score.keys():
						sentence_score[sentence] = text_dtm[word]
					else:
						sentence_score[sentence] += text_dtm[word]

	return sentence_score


#use first n-th highest scoring sentences as summary
def summarize(sentence_score):
	summary = heapq.nlargest(SUMMARY_LENGTH, sentence_score, key=sentence_score.get)
	summary = ' '.join(summary)

	return summary


text = getText(URL)
text = combineText(text)

#Clean text
text = cleanText_round1(text)

text_clean = cleanText(text)

#Convert to sentences
sentences = nltk.sent_tokenize(text)

text_dtm = countVectorize(text_clean)
sentence_score = sentenceScore(sentences, text_dtm)


print("Original Text\n\n",text)
print("\nSummary\n\n",summarize(sentence_score))








