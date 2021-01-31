'''
Created on Apr 23, 2014

@author: Girish
'''
"""import math

cache_enabled = False
idf_cache = {}

def cache_enabled(enabled):
    cache_enabled = enabled
    if enabled:
        clear_idf_cache()
    

def clear_idf_cache():
    idf_cache.clear()
  

def freq(word, doc):
    return doc.count(word)


def word_count(doc):
    return len(doc)


def tf(word, doc):
    fr = freq(word, doc)
    if fr > 0:
        return (fr / float(word_count(doc)))
#    return 1
    else:
        return 0


def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    if cache_enabled and idf_cache.has_key(word):
        return idf_cache.get(word)
  
    else:
        idfv = math.log(len(list_of_docs) /float(num_docs_containing(word, list_of_docs)))
    
    if cache_enabled:
        idf_cache[word] = idfv
    
    return idfv


def tf_idf(word, doc, list_of_docs):
    tfv = tf(word, doc)
    if tfv > 0:
        return (tfv * idf(word, list_of_docs))
    else:
        return 0
        """
import math

cache_enabled = False
idf_cache = {}

def cache_enabled(enabled):
    cache_enabled = enabled
    if enabled:
        clear_idf_cache()
    

def clear_idf_cache():
    idf_cache.clear()


def tf(word, doc):
    fr = doc.count(word)
    if fr > 0:
        return (fr / float(len(doc)))
#    return 1
    else:
        return 0


def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if document.count(word) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    if cache_enabled and idf_cache.has_key(word):
        return idf_cache.get(word)
  
    else:
        idfv = math.log(len(list_of_docs) /float(num_docs_containing(word, list_of_docs)))
    
    if cache_enabled:
        idf_cache[word] = idfv
    
    return idfv


def tf_idf(word, doc, list_of_docs):
    tfv = tf(word, doc)
    if tfv > 0:
        return (tfv * idf(word, list_of_docs))
    else:
        return 0