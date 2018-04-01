import sys
import numpy as np
import tensorflow as tf
import re
from pymongo import MongoClient
import json
import time
from datetime import datetime, timedelta
import fnmatch
import os


numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore(sess, tf.train.latest_checkpoint('models'))

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix



#inputText = "The best phone u get at this range. Nice camera. 2GB RAM. Latest update. Well u can never get a phone better than this at this range. It has several nice features. Internal memory is also quite good with 16 GB. processor is also doing quite good. Do believe me and by this one. u will have no issues for sure."
#inputMatrix = getSentenceMatrix(inputText)


def writeIntoDb(data, level):
    con = MongoClient('localhost', 27017)
    db= con.sanlp
    #rec_id=reviews.insert_one(data)
    if level == 0:
        rec_id = db.sentimentAnalysis.insert_one(data)
        print(rec_id)
    elif level == 1:
        rec_id = db.reviewText.insert_one(data)
        print(rec_id)

    con.close()

def computeReviewSentiment(reviewList, dir, timestr):
    for s in review_files:
        file = dir+"/"+s
        print(file)
        f = open(file)
        line = f.readline()
        positiveCount =0
        negativeCount =0
        totalCount=0
        while line:
            try:
                if line == '[' or line == ']':
                    break
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1]
                parsed_json = json.loads(line)
                review_data = parsed_json['review_text']
                product_name = parsed_json['product_name']
                product_id = parsed_json['product_id']
                print(product_name)
                print(product_id)
                inputMatrix = getSentenceMatrix(review_data)
                predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
                # predictedSentiment[0] represents output score for positive sentiment
                # predictedSentiment[1] represents output score for negative sentiment
                print (review_data)
                totalCount=totalCount+1
                print(totalCount)
                if (predictedSentiment[0] > predictedSentiment[1]):
                    print (" ******* Positive Sentiment")
                    positiveCount = positiveCount + 1
                    rev_sentiment = {'p_id' : product_id, 'review_text': review_text, 'sentiment' : 1, 'date':timestr}
                    writeIntoDb(rev_sentiment,1)

                else:
                    print (" ******* Negative Sentiment")
                    negativeCount = negativeCount +1
                    rev_sentiment = {'p_id' : product_id, 'review_text': review_text, 'sentiment' : 0, 'date':timestr}
                    writeIntoDb(rev_sentiment,1)
                #line = f.readline()

            except Exception as e:
                print(e)
                print(line)
            line = f.readline()
        f.close()
        print(negativeCount)
        print(positiveCount)
        print(totalCount)
        if totalCount!= 0:
            negativeSentiment = float(negativeCount)/float(totalCount)
            positiveSentiment = float(positiveCount)/float(totalCount)
            print(float(negativeSentiment))
            rev_data = {'p_id' : product_id, 'p_name': product_name, 'date':timestr, 'sentiment' : { 'positive' : positiveSentiment, 'negative' : negativeSentiment}}

            print(rev_data)
            writeIntoDb(rev_data,0)


def sAnalysis(reviewLogPath, timestr):
    print(timestr)
    review_files = []
    rf_count=0
    #dir='/opt/src/sanlp/azcrawl/Reviews'
    #dir = '/opt/src/sanlp/Fk_Crawler/reviews'
    print reviewLogPath
    for file in os.listdir(az):
        if fnmatch.fnmatch(file, 'review-'+timestr+'*.log'):
            print(file)
            #product_id , review, sentiment, date
            review_files.append(file)

    computeReviewSentiment(review_files, dir, timestr)


def main():

    global az
    global fk

    parser = argparse.ArgumentParser(description="Sentiment analysis help")
    parser.add_argument("--az", help="The datacenter (i.e. tpa1 ,tea2 etc.)", required=True)
    parser.add_argument("--fk", help="The environment (i.e. latest , integration etc.)", required=True)

    args = parser.parse_args()
    timestr = datetime.strftime(datetime.now() - timedelta(1), '%d_%m_%Y')
    if args.az:
        az = args.az
        sAnalysis(az, timestr)

    if args.fk:
        fk = args.fk
        sAnalysis(fk, timestr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print "Exception running script: ", str(e)
        sys.exit(1)
