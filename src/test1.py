'''
Created on Apr 22, 2014

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

POS_Words=[]
NEG_Words=[]
MIX_Words=[]
NEUTRAL_Words=[]

def evaluate_features(feature_select):
    
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
    
    tempPos=[]   
    
    stopWords = stopwords.words("english")
              
    # Reading positive words from txt file
    fileInput = open('positive-words.txt', 'r')
    sentences = re.split(r'\n', fileInput.read())
    fileInput.close()
       
    for i in sentences: 
        posWords=re.findall(r"^[\w']+",i)
        if posWords:
            posWords = [feature_select(posWords), '+']
            POS_Words.append(posWords)
            pos_Feautures.append(posWords)
    
    # Reading negative words from txt file
    fileInput = open('negative-words.txt', 'r')
    sentences = re.split(r'\n', fileInput.read())
    fileInput.close()
    for i in sentences:
        negWords=re.findall(r"^[\w']+",i)
        if negWords:
            negWords = [feature_select(negWords), '-']
            NEG_Words.append(negWords)
            neg_Feautures.append(negWords)
        
    #reading pre-labeled input and splitting into lines
    fileInput = open('All_Classified.txt', 'r')
    sentences = re.split(r'\n', fileInput.read())
    fileInput.close()
    
    for i in sentences:
        #tagged = re.findall(r"^[-=+\*]|[\w']+[/]?[\w']+[/]+[\w']+ [.,!?;]*", i)
        tagged = re.findall(r"^[-=+\*]|[\w']+[/]?[\w']+[/]+[^(NN|NNS|NNP|PRP)]+ [.,!?;]*", i)
        untagged =re.sub(r'/[^\s]+|[0-9]+|[.,!?;]*|','',' '.join(tagged))
              
        untagged_Words= re.findall(r"[\w']+|[.,!?;]", untagged)
        filtered_Words = [w for w in untagged_Words if not w.lower() in stopWords]
        
                
        if untagged  and  tagged:
            if untagged[0]=='-':
                neg_sentence.append(untagged)
                filtered_Words = [feature_select(filtered_Words), '-']
                NEG_Words.append(filtered_Words)
                neg_Feautures.append(filtered_Words)
                
                tagged_Words = [feature_select(tagged), '-']
                NEG_Words.append(tagged_Words)
                neg_Feautures.append(tagged_Words)
                """                
                for word in filtered_Words:
                    a=dict([(word, True)])
                    Word= [a, '-'] 
                    NEG_Words.append(Word)
                    neg_Feautures.append(Word)            
                 """   
            if untagged[0]=='+':
                pos_sentence.append(untagged)
                
                filtered_Words = [feature_select(filtered_Words), '+']
                POS_Words.append(filtered_Words)
                pos_Feautures.append(filtered_Words)
                
                tagged_Words = [feature_select(tagged), '+']
                POS_Words.append(tagged_Words)
                pos_Feautures.append(tagged_Words)
                """
                for word in filtered_Words:
                    a=dict([(word, True)])
                    Word= [a, '+'] 
                    #Word= [dict([(word, True)]), '+'] 
                    POS_Words.append(Word)
                    pos_Feautures.append(Word)    
                """
            if untagged[0]=='*':
                mixed_sentence.append(untagged)
                filtered_Words = [feature_select(filtered_Words), '*']
                MIX_Words.append(filtered_Words)
                mixed_Feautures.append(filtered_Words)
                                       
            if untagged[0]=='=':
                neutral_sentence.append(untagged)                
                filtered_Words = [feature_select(filtered_Words), '=']
                NEUTRAL_Words.append(filtered_Words)
                neutral_Feautures.append(filtered_Words)
                """
                tagged_Words = [feature_select(tagged), '=']
                NEUTRAL_Words.append(tagged_Words)
                neutral_Feautures.append(tagged_Words)
                
                for word in filtered_Words:
                    a=dict([(word, True)])
                    Word= [a, '='] 
                    #Word= [dict([(word, True)]), '='] 
                    NEUTRAL_Words.append(Word)
                    neutral_Feautures.append(Word) 
               """                          
        tagged_Sentences.append(tagged)
        untagged_Sentences.append(untagged)
      
    
    #Read a test file and create test feutures
     
    fileInput = open('test_dummy.txt', 'r')
    sentences = re.split(r'\n', fileInput.read())
    fileInput.close()
    
    for i in sentences:
        tagged = re.findall(r"^[-=+\*]|[\w']+[/]?[\w']+[/]+[\w']+ [.,!?;]*", i)
        untagged =re.sub(r'/[^\s]+|[0-9]+|[.,!?;]*','',i)
       
        untagged_Words= re.findall(r"[\w']+|[.,!?;]", untagged)
        filtered_Words = [w for w in untagged_Words if not w.lower() in stopWords]
        
        if untagged  and  tagged:
            if untagged[0]=='-': 
                test_sentence.append(untagged)
                filtered_Words = [feature_select(filtered_Words), '-']
                test_Feautures.append(filtered_Words)
            if untagged[0]=='+': 
                test_sentence.append(untagged)
                filtered_Words = [feature_select(filtered_Words), '+']
                test_Feautures.append(filtered_Words)          
            if untagged[0]=='=':
                test_sentence.append(untagged)
                filtered_Words = [feature_select(filtered_Words), '=']
                test_Feautures.append(filtered_Words)
                  
    #posCutoff = int(math.floor(len(pos_Feautures)*3/4))
    #negCutoff = int(math.floor(len(neg_Feautures)*3/4))
    neutralCutoff = int(math.floor(len(neutral_Feautures)*1/20))
    
    trainFeatures = pos_Feautures + neg_Feautures + neutral_Feautures[:neutralCutoff]    
       
    
    #test_Feautures= pos_Feautures[posCutoff:] + neg_Feautures[negCutoff:] + neutral_Feautures[neutralCutoff: 2*neutralCutoff]
    #trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)
    
    #initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)    

    #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(test_Feautures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)    

    #prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (len(tagged_Sentences), len(test_sentence))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, test_Feautures)
    
    print 'pos precision:', nltk.metrics.precision(referenceSets['+'], testSets['+'])
    print 'pos recall:', nltk.metrics.recall(referenceSets['+'], testSets['+'])
    print 'pos f-measure:',nltk.metrics.f_measure(referenceSets['+'], testSets['+'])
    
    print 'neg precision:', nltk.metrics.precision(referenceSets['-'], testSets['-'])
    print 'neg recall:', nltk.metrics.recall(referenceSets['-'], testSets['-'])
    print 'neg f-measure:',nltk.metrics.f_measure(referenceSets['-'], testSets['-'])
    
    #classifier.show_most_informative_features(10)
   
    
#creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])


#scores words based on chi-squared test to show information gain 
def create_word_scores():
    
    global POS_Words
    global NEG_Words
    global MIX_Words
    global NEUTRAL_Words
    
    POS_Words = list(itertools.chain(*POS_Words))
    NEG_Words = list(itertools.chain(*NEG_Words))
    MIX_Words = list(itertools.chain(*MIX_Words))
    NEUTRAL_Words = list(itertools.chain(*NEUTRAL_Words))

    #build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    
    for word in POS_Words:
        for key in word:
            word_fd.inc(key.lower())
            cond_word_fd['+'].inc(key.lower())
    for word in NEG_Words:
        for key in word:
            word_fd.inc(key.lower())
            cond_word_fd['-'].inc(key.lower())
    for word in MIX_Words:
        for key in word:
            word_fd.inc(key.lower())
            cond_word_fd['*'].inc(key.lower())
    for word in NEUTRAL_Words:
        for key in word:
            word_fd.inc(key.lower())
            cond_word_fd['='].inc(key.lower())
    

    #finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['+'].N()
    neg_word_count = cond_word_fd['-'].N()
    #mix_word_count = cond_word_fd['*'].N()
    neutral_word_count = cond_word_fd['='].N()
   # total_word_count = pos_word_count + neg_word_count + mix_word_count+ neutral_word_count 
    total_word_count = pos_word_count + neg_word_count+neutral_word_count
    
    #builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['+'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['-'][word], (freq, neg_word_count), total_word_count)
        #mix_score = BigramAssocMeasures.chi_sq(cond_word_fd['*'][word], (freq, mix_word_count), total_word_count)
        neutral_score = BigramAssocMeasures.chi_sq(cond_word_fd['='][word], (freq, neutral_word_count), total_word_count)
        
        #word_scores[word] = pos_score + neg_score + mix_score+neutral_score
        word_scores[word] = pos_score + neg_score + neutral_score

    return word_scores


#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


"""
Main execution starts from here

"""
evaluate_features(make_full_dict)

#finds word scores
word_scores = create_word_scores()

#numbers of features to select
#numbers_to_test = [10,100,1000,10000]
numbers_to_test = [10,100]
#tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
    print '\n\nEvaluating best %d word features ' % (num)
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features) 

    """
# Divide the data for training and testing       
        sizeOfValidationSet= int(math.floor(len(untagged_Sentences)*1/10))
        validationSet=[]
        startIndex=1
        endIndex=sizeOfValidationSet

    for i in range(0,10):
        if(endIndex <= len(untagged_Sentences)):
            validationSet.append(untagged_Sentences[startIndex:endIndex])
            startIndex=startIndex+sizeOfValidationSet+1
            endIndex=endIndex+sizeOfValidationSet

    trainingSet=[]
    testingSet=[]
    tempTrainSet=[]

    #for i in range(0,10):
    iRand = randrange(0, 10,1)
    testingSet=validationSet[iRand]
    trainingSet=validationSet[:iRand]+validationSet[iRand+1:]

    """
 
    
    

         

