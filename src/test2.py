'''
Created on Apr 22, 2014

@author: Girish
'''
import sys
import os
import re
import nltk
from nltk.corpus import wordnet
from replacer import RegexpReplacer
from replacer import RepeatReplacer

def main():
    # Read the command line arguments
    args = sys.argv[1:]
    if not args:
        _exit()
    
    # For holding training feature set
    train_features = [] 
    
    # Validate the file 
    for infile in args:
        if not infile or not os.path.isfile(infile):
            print "File doesn't exist", infile
            continue
    
    # Declare lists to be populated with data
    tweets = []
    classes = []
    
    # Read the input file and populate the lists  
    f = open(infile, 'rU')
    for line in f.readlines()[1:]:
        lineparts = line.split(',')
        if isInteger(lineparts[-1]):
            tweets.append(' '.join(lineparts[:-1]))
            classes.append(int(lineparts[-1]))
    f.close()
   
  #  tweets = ["obama said he believe romney is a good man and 
  #  loves his family then he wanted to say but he don't give a fuck about america lol"]
  #  tweets = tweets[0:50]
    original_tweets = [tweet for tweet in tweets]
    
    tweets = clean(tweets)
    tweets = stem(tweets)
    tweets = remove_stop_words(tweets)
    tweets = replace_entities(tweets)
    
    for i in range(len(tweets)):
        print original_tweets[i]
        print tweets[i]
        print
      
    # Tokenize the tweets for further processing
    tokenized_tweets = [tweet.split() for tweet in tweets]
    del tweets
    
    # Add bigrams to the tokenized tweets
    from nltk import bigrams
    tokenized_tweets = [tweet + [bigram[0]+'_'+bigram[1] for bigram in bigrams(tweet)] for tweet in tokenized_tweets]
    #  print '\n'.join([str(tweet) for tweet in tokenized_tweets])
    
    # Extract the features from training data (This is done only for the trainfile)
    if not train_features:
        train_features = get_word_features(tokenized_tweets)
        print "Training data has", len(train_features), "features"
    
    # List of features for current data file
    features = [feature for feature in train_features]
    
    # Create the classifier model from feature list
    classifier_model = generate_classifier_model(tokenized_tweets, features)
    # print '\n'.join([str(t_model) for t_model in classifier_model])
    
    # Add the sentiment score to the features and classifier model
    add_sentiment_score(tokenized_tweets, features, classifier_model)
    
    # Append the classes to the features and the classifier model
    features.append('final_class')
    class_index = len(features) -1
    for i,t_model in enumerate(classifier_model):
        t_model[class_index] = classes[i]
    
    outfile = os.path.join(os.path.abspath(os.path.join(infile, os.pardir)), "model_" + os.path.basename(infile).split('.')[0] + ".arff")
    print "Writing model to", outfile  
    write_model_to_arff(features, classifier_model, outfile)
                            

def clean(sentences):
    """Remove unwanted characters, tags and patterns from sentences"""
    import string
    

    # Compile all the regex patterns
    ae_tags = re.compile(r'<[/ae]+>')
    url = re.compile(r'http://\S+')
    at_tag = re.compile(r'[@#]\S+')
    punctuation = re.compile(r'[\[\]\'!"#$%&\\()*+,-./:;<=>?@^_`{}~|0-9]')
    multispace = re.compile(r'\s+')

    sentences = [filter(lambda x: x in string.printable, tweet).lower() for tweet in sentences]   # Remove the non-printable characters and covert string to lower case
    sentences = [ae_tags.sub(' ',tweet) for tweet in sentences]   # Remove the entity and aspect tags
    sentences = [url.sub('',tweet) for tweet in sentences]    # Remove the urls
    sentences = [at_tag.sub('',tweet) for tweet in sentences]   # Remove the annotation and hash tags
  
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
  
#  f = open('../resource/stopwords.txt', 'rU')
#  stop_words = f.read().split('\n')
#  f.close()
#  exclusion_list = []
  
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


def replace_entities(tweets):
    mitt_pat = re.compile(r'\bm+i+t+[romneystr]*\b')
    romney_pat = re.compile(r'\br+o+m+[neysao]*\b')
    barack_pat = re.compile(r'\bb+a+r+a+c+[kobama]*\b')
    obama_pat = re.compile(r'\bo+b+a+[mbaswohie]*\b')
  
    tweets =  [mitt_pat.sub('romney', tweet) for tweet in tweets]
    tweets =  [romney_pat.sub('romney', tweet) for tweet in tweets]
    tweets =  [barack_pat.sub('obama', tweet) for tweet in tweets]
    tweets =  [obama_pat.sub('obama', tweet) for tweet in tweets]
  
    romney_pat = re.compile('romney\sromney')
    obama_pat = re.compile('obama\sobama')
  
    tweets =  [romney_pat.sub('romney', tweet) for tweet in tweets]
    tweets =  [obama_pat.sub('obama', tweet) for tweet in tweets]
  
    return tweets 
  

def add_sentiment_score(sentences, features, model):
    f = open('../resource/positive-words.txt', 'rb')
    pos_words = f.read().split('\n')
    f.close()
  
    f = open('../resource/negative-words.txt', 'rb')
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


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  


def get_word_features(tokenized_tweets):
    wordlist = get_words_in_tweets(tokenized_tweets)
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


def get_word_features_from_file(f):
    f = open(f, 'rU')
    wordlist = f.read().split('\n')
    f.close()
    print wordlist
    return wordlist


def write_features_to_file(features, f):
    f = open(f, 'w')
    f.write('\n'.join(features))
    f.close()


def generate_csv_classifier_model(tokenized_tweets, features):
    model = []
  
    import tfidf
    tfidf.cache_enabled(True)
  
    for tweet in tokenized_tweets:
        tweet_model = []
        for feature in features:
            tweet_model.append(tfidf.tf_idf(feature, tweet, tokenized_tweets))
#      if feature in tweet:
#        tweet_model.append(1)
#      else:
#        tweet_model.append(0)
    
    model.append(tweet_model)
    return model


def generate_classifier_model(tokenized_sentences, features):
    model = []
  
    import tfidf
    tfidf.cache_enabled(True)
  
    for tweet in tokenized_sentences:
        tweet_model = {}
        for i,feature in enumerate(features):
            tfidfv = tfidf.tf_idf(feature, tweet, tokenized_sentences)
            if tfidfv > 0:
                tweet_model[i] = tfidfv
      
#      if feature in tweet:
#        tweet_model[i] = 1
    
    model.append(tweet_model)
    return model


def write_model_to_csv(word_list, model, outfile):
    text = ','.join(word_list) + '\n'
    text = text + '\n'.join([','.join([str(val) for val in t_model]) for t_model in model])
#  print text
  
    f=open(outfile, 'w')
    f.write(text)
    f.close()

  
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
  

def isInteger(i):
    try:
        int(i)
        return True
    except ValueError:
        return False


def _exit(message=None):
    if message == None:
        message = "usage: <module> trainfile [testfile...]"
        print message
    sys.exit(1)


if __name__ == '__main__':
    main()
