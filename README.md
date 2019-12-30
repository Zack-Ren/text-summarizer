# text-summarizer
Summarize articles or long web pages with NLTK 

### Summarizing text with NLTK in Python
1. Scrape traget website for text
2. Join list of sentences to one large paragraph
3. Preprocess data
    1. Convert to lowercase
    2. Remove brackets
    3. Remove new-line (\n) characters
    4. Remove words containing numbers
    5. Remove punctuation
4. Tokenize sentences (from un-processed data)
5. Find weighted frequency of each word occurance
6. Calculate weight of each sentence
7. Summarize with n-th highest weighted sentences



