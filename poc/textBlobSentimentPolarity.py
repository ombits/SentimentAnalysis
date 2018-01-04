from textblob import TextBlob


blob = TextBlob(open("/Users/osingh/Desktop/Data-Science/Sentiment-Analysis/Review_data/few.json").read())
for word in blob.words:
    print word


blob.tags

blob.noun_phrases

for sentence in blob.sentences:
    print(sentence.sentiment)
    print(sentence.sentiment.polarity)
