# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:12:41 2020

@author: Jörg
"""

import pandas as pd

completeData = pd.read_csv("stackexchange_812k.csv", dtype={'post_id': str, 'parent_id' : str, 'comment_id' : str})
completeData.head()
completeData.columns
#####
pd.set_option('display.max_colwidth', -1)
#####


filter500 =  (completeData["text"].str.len()  > 0) &  (completeData["text"].str.len()  < 500 )
filteredCompleteData = completeData[filter500]


####

postData = filteredCompleteData[filteredCompleteData.category == "post"]

titleData = filteredCompleteData[filteredCompleteData.category == "title"]

commentData = filteredCompleteData[filteredCompleteData.category == "comment"]




### extraact html tags 

from bs4 import BeautifulSoup
import re


########
regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
latexregex = r'\$.*\$'
endregex = r'\n'
numberregex = r'[0-9]'
sonderregex = r'[\(\)"\*%\[\]\-\']'
whiteSpaceReduceRegex = r'\s\s+'
refregex = r'@.*'
######

def preparePost(df):
    df = df.applymap(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    df = df.applymap(lambda text: re.sub(regex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(latexregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(endregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(numberregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(sonderregex, '', text, flags=re.MULTILINE))
    return df
    
    
####
postDataText =  postData.loc[:,["text"]]    
postData.loc[:,"texttokens" ] =   preparePost(postDataText).text
#######

def prepareComment(df):
    df = df.applymap(lambda text: re.sub(regex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(latexregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(endregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(numberregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(sonderregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(refregex, '', text, flags=re.MULTILINE))
    return df

commentDataText =  commentData.loc[:,["text"]]
commentData.loc[:,"texttokens" ] =   prepareComment(commentDataText).text


######
def prepareTitle(df):
    df = df.applymap(lambda text: re.sub(numberregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(sonderregex, '', text, flags=re.MULTILINE))
  ##  df = df.applymap(lambda text: re.sub(whiteSpaceReduceRegex, ' ', text, flags=re.MULTILINE))
    return df

titleDataText =  titleData.loc[:,["text"]]

titleData.loc[:,"texttokens" ] =   prepareTitle(titleDataText).text


#####
import nltk
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

def tokenizeAll(text, tokenizer):
    tempList = nltk.sent_tokenize(text)    
    startStr = ''
    for sentence in tempList:
        res = tokenizer.tokenize(sentence, return_str=True).lower()       
        startStr = startStr + res
    startStr = re.sub(whiteSpaceReduceRegex, ' ', startStr, flags=re.MULTILINE) 
    return startStr.strip()
     
##res = tokenizeAll(teststr,tokenizer)    

#########################################
### tokenize
    
df =  commentData.loc[:,["texttokens"]]
commentData.loc[:,"texttokens" ] =  df.applymap(lambda text: tokenizeAll(text, tokenizer)) 

df =  postData.loc[:,["texttokens"]]
postData.loc[:,"texttokens" ] =  df.applymap(lambda text: tokenizeAll(text, tokenizer)) 

df =  titleData.loc[:,["texttokens"]]
titleData.loc[:,"texttokens" ] =  df.applymap(lambda text: tokenizeAll(text, tokenizer)) 

newDF = pd.concat([commentData, postData, titleData])

newDF = newDF.loc[ newDF["texttokens"].str.len()  > 20, :]
newDF.columns

import csv
newDF.to_csv('JSCH1.csv', index=False,  quoting = csv.QUOTE_MINIMAL)

testdf = pd.read_csv('JSCH1.csv')

testdf.head()


