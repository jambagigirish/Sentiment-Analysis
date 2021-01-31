'''
Created on Apr 23, 2014

@author: Girish
'''
import os
import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from random import randrange
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from replacer import RegexpReplacer
from replacer import RepeatReplacer

POS_Words=[]
NEG_Words=[]
MIX_Words=[]
NEUTRAL_Words=[]

def evaluate_features():
    
    #All variables
    tagged_Sentences = []
    untagged_Sentences=[]
    neg_sentence=[]
    pos_sentence=[]
    mixed_sentence=[]
    neutral_sentence=[]
    neg_Feautures=[]
    pos_Feautures=[]
    mixed_Feautures=[]
    neutral_Feautures=[]
    test_sentence=[]
    test_Feautures=[]
    allwords=[]
    
    train_features = []  
    stopWords = stopwords.words("english")   
    newsentences = []
    Newsentences = []
    classes = []
    test_classes = []
    infile='All_Classified.txt'
    #infile='test_dummy.txt'
    testfile='cs583_test_data.txt'
    temp=0
    
    #reading pre-labeled input and splitting into lines
    fileInput = open(infile, 'r')
    sentences = re.split(r'\n', fileInput.read())
    fileInput.close()
    for i in sentences:
        tagged = re.findall(r"^[012(-1)]+|[\w']+[/]?[\w']+[/]+[\w']+ [.,!?;]*", i)
        #tagged = re.findall(r"^[-=+\*]|[\w']+[/]?[\w']+[/]+[^(NN|NNS|NNP|PRP)]+ [.,!?;]*", i)
        untagged =re.sub(r'/[^\s]+|[0-9]+|[.,!?;]*|','',' '.join(tagged))
        #untagged =re.sub(r'/[^\s]+|[0-9]+|[.,!?;]*|','',i)
              
        untagged_Words= re.findall(r"[\w']+|[.,!?;]", untagged)
        #filtered_Words = [w for w in untagged_Words if not w.lower() in stopWords]
                       
        if untagged  and  tagged:
            if tagged[0]=='0':
                if temp<=2000:
                    newsentences.append(untagged[1:])
                    classes.append(tagged[0])
                    temp=temp+1
            if tagged[0]=='1' or tagged[0]=='2':
                newsentences.append(untagged[1:])
                classes.append(tagged[0])  
            if tagged[0]=='-1':
                newsentences.append(untagged[1:])
                classes.append(tagged[0])                        
                   
    original_sentences = [s for s in newsentences]
    
    newsentences = clean(newsentences)
    newsentences = stem(newsentences)
    newsentences = remove_stop_words(newsentences)
        
    tokenized_sentences = [s.split() for s in newsentences]
    del newsentences
    
    # Add bigrams to the tokenized tweets
    from nltk import bigrams
    tokenized_sentences = [s + [bigram[0]+'_'+bigram[1] for bigram in bigrams(s)] for s in tokenized_sentences]

    # Extract the features from training data (This is done only for the trainfile)
    if not train_features:
        train_features = get_word_features(tokenized_sentences)  
        print "Training data has", len(train_features), "features"
    
    # List of features for current data file
    features = [feature for feature in train_features]
              
    # Create the classifier model from feature list
    classifier_model = generate_classifier_model(tokenized_sentences, features)
    #  print '\n'.join([str(t_model) for t_model in classifier_model])
    
    # Add the sentiment score to the features and classifier model
    add_sentiment_score(tokenized_sentences, features, classifier_model)
    
    features.append('final_class')
    class_index = len(features) -1
    for i,t_model in enumerate(classifier_model):
        t_model[class_index] = classes[i]
    
    outfile = os.path.join(os.path.abspath(os.path.join(infile, os.pardir)), "model_" + os.path.basename(infile).split('.')[0] + ".arff")
    print "Writing model to", outfile  
    write_model_to_arff(features, classifier_model, outfile)
   
    ############################test file
    
    #reading pre-labeled input and splitting into lines
    fileInput = open(testfile, 'r')
    sentences = re.split(r'\n', fileInput.read())
    fileInput.close() 
    
    for i in sentences:
        tagged = re.findall(r"^[012\-1]|[\w']+[/]?[\w']+[/]+[\w']+ [.,!?;]*", i)
        #tagged = re.findall(r"^[-=+\*]|[\w']+[/]?[\w']+[/]+[^(NN|NNS|NNP|PRP)]+ [.,!?;]*", i)
        untagged =re.sub(r'/[^\s]+|[0-9]+|[.,!?;]*|','',' '.join(tagged))
        #untagged =re.sub(r'/[^\s]+|[0-9]+|[.,!?;]*|','',i)
              
        untagged_Words= re.findall(r"[\w']+|[.,!?;]", untagged)
        #filtered_Words = [w for w in untagged_Words if not w.lower() in stopWords]
                       
        if untagged  and  tagged and i:
            if i[-1]=='0':
                if temp<=2000:
                    Newsentences.append(untagged)
                    test_classes.append(i[-1])
                    temp=temp+1
            if i[-1] == '1' or i[-2] == '-' or i[-1] == '2' :
                Newsentences.append(untagged)
                if i[-2] == '-':
                    test_classes.append('-1')
                else: test_classes.append(i[-1])
                                        
        #tagged_Sentences.append(tagged)
        #untagged_Sentences.append(untagged)
           
    original_sentences = [s for s in Newsentences]
    
    Newsentences = clean(Newsentences)
    Newsentences = stem(Newsentences)
    Newsentences = remove_stop_words(Newsentences)
    
    
    tokenized_sentences = [s.split() for s in Newsentences]
    del Newsentences
    
    # Add bigrams to the tokenized tweets
    from nltk import bigrams
    tokenized_sentences = [s + [bigram[0]+'_'+bigram[1] for bigram in bigrams(s)] for s in tokenized_sentences]

    # Extract the features from training data (This is done only for the trainfile)
    if not train_features:
        train_features = get_word_features(tokenized_sentences)  
        print "Training data has", len(train_features), "features"
    
    # List of features for current data file
    features = [feature for feature in train_features]
              
    # Create the classifier model from feature list
    classifier_model = generate_classifier_model(tokenized_sentences, features)
    #  print '\n'.join([str(t_model) for t_model in classifier_model])
    
    # Add the sentiment score to the features and classifier model
    add_sentiment_score(tokenized_sentences, features, classifier_model)
    
    features.append('final_class')
    class_index = len(features) -1
    for i,t_model in enumerate(classifier_model):
        t_model[class_index] = test_classes[i]
    
    outfile = os.path.join(os.path.abspath(os.path.join(testfile, os.pardir)), "model_" + os.path.basename(testfile).split('.')[0] + ".arff")
    print "Writing model to", outfile  
    write_model_to_arff(features, classifier_model, outfile)
    


################################################################
   
def clean(sentences):
    """Remove unwanted characters, tags and patterns from sentences"""
    import string
    
    # Compile all the regex patterns
    punctuation = re.compile(r'[\[\]\'!"#$%&\\()*+,-./:;<=>?@^_`{}~|0-9]')
    multispace = re.compile(r'\s+')

    sentences = [word.lower() for word in sentences]   # Remove the non-printable characters and covert string to lower case
     
    # Replace the common apostophe patterns like 's, 'll etc.
    replacer = RegexpReplacer()
    sentences = [replacer.replace(tweet) for tweet in sentences]
  
    sentences = [punctuation.sub(' ',tweet) for tweet in sentences]   # Remove all the punctuation marks
    sentences = [multispace.sub(' ',tweet).strip() for tweet in sentences]    # Remove leading,trailing,multiple spaces
    replacer = RepeatReplacer()
    sentences = [replacer.replace(tweet) for tweet in sentences]
    
    return sentences

def remove_stop_words(tweets):
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    exclusion_list = ['no', 'not', 'nor']
  
    # Remove the stopwords
    tweets = [' '.join([w for w in tweet.split() if w in exclusion_list or (len(w) > 2 and not w in stop_words)]) for tweet in tweets]
    
    return tweets


def stem(tweets):
    from nltk.stem.wordnet import WordNetLemmatizer
    global lmtzr
    lmtzr = WordNetLemmatizer()
  
    tweets = [lemmatize(tweet) for tweet in tweets]
    return tweets


def lemmatize(tweet):
    tokens = tweet.split()
    #  tokens = nltk.pos_tag(tokens)
    #  tokens = [lmtzr.lemmatize(word, get_wordnet_pos(tag)) for (word,tag) in tokens]
  
    for i in range(len(tokens)):
        res = lmtzr.lemmatize(tokens[i])
        if res == tokens[i]:
            tokens[i] = lmtzr.lemmatize(tokens[i], 'v')
        else:
            tokens[i] = res
            
    return ' '.join(tokens)


def add_sentiment_score(sentences, features, model):
    f = open('positive-words.txt', 'rb')
    pos_words = f.read().split('\n')
    f.close()
    
  
    f = open('negative-words.txt', 'rb')
    neg_words = f.read().split('\n')
    f.close()
  
    features.append('pos_count')
    features.append('neg_count')
    pos_count_index = len(features) - 2
    neg_count_index = len(features) - 1
    for i,tweet in enumerate(sentences):
        pos_count=0
        neg_count=0
        for word in tweet:
            if word in pos_words:
                pos_count = pos_count + 1
                continue
            if word in neg_words:
                neg_count = neg_count + 1
            if pos_count > 0:
                model[i][pos_count_index] =  pos_count
            if neg_count > 0:
                model[i][neg_count_index] =  neg_count
    return model


def get_word_features(tokenized_sentences):
    wordlist = get_words_in_tweets(tokenized_sentences)
    wordlist = nltk.FreqDist(wordlist)
    word_features = get_frequent_words(wordlist.items(), 0)
    return word_features
 
def get_words_in_tweets(tokenized_tweets):
    all_words = []
    for tweet in tokenized_tweets:
        all_words.extend(tweet)
    return all_words

def get_frequent_words(wordlist, freq_threshold):    
    return [word for word,freq in wordlist if freq >= freq_threshold]   

def generate_classifier_model(tokenized_sentences, features):
    model = []
  
    import tfidf
    tfidf.cache_enabled(True)
  
    for sentence in tokenized_sentences:
        tweet_model = {}
        for i,feature in enumerate(features):
            tfidfv = tfidf.tf_idf(feature, sentence, tokenized_sentences)
            
            if tfidfv > 0:
                tweet_model[i] = tfidfv
        model.append(tweet_model)
    return model

def write_model_to_arff(word_list, model, outfile):
    lines = []
    lines.append("@RELATION opinion")
    lines.append("\n")
    lines = lines + ["@ATTRIBUTE "+word+" NUMERIC" for word in word_list[:-1]]
    lines = lines + ["@ATTRIBUTE "+word_list[-1]+" {-1,0,1,2}"]
    lines.append("\n")
    lines.append("@DATA")
    for t_model in model:
        lines.append('{'+ ','.join([str(k)+' '+str(t_model[k]) for k in sorted(t_model.keys())]) + '}')
  
    f=open(outfile, 'w')
    f.write('\n'.join(lines))
    f.close() 
       

evaluate_features()        
