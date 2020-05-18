# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:12:41 2020

@author: Jörg
"""

import pandas as pd

completeData = pd.read_csv("stackexchange_812k.csv")
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
sonderregex = r'[\(\)"\*%\[\]\-]'
whiteSpaceReduceRegex = r'\s\s+'
refregex = r'@.*'
######


test = postData.loc[:,["text"]].applymap(lambda text: len(text))

postDataText =  postData.loc[:,["text"]]

def preparePost(df):
    df = df.applymap(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    df = df.applymap(lambda text: re.sub(regex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(latexregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(endregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(numberregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(sonderregex, '', text, flags=re.MULTILINE))
   ## df = df.applymap(lambda text: re.sub(whiteSpaceReduceRegex, ' ', text, flags=re.MULTILINE))
    return df
    
    
####
postData.loc[:,"text" ] =   preparePost(postDataText)
#######

def prepareComment(df):
    df = df.applymap(lambda text: re.sub(regex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(latexregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(endregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(numberregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(sonderregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(refregex, '', text, flags=re.MULTILINE))
   ##df = df.applymap(lambda text: re.sub(whiteSpaceReduceRegex, ' ', text, flags=re.MULTILINE))
    return df

commentDataText =  commentData.loc[:,["text"]]
commentData.loc[:,"text" ] =   prepareComment(commentDataText)


######
def prepareTitle(df):
    df = df.applymap(lambda text: re.sub(numberregex, '', text, flags=re.MULTILINE))
    df = df.applymap(lambda text: re.sub(sonderregex, '', text, flags=re.MULTILINE))
  ##  df = df.applymap(lambda text: re.sub(whiteSpaceReduceRegex, ' ', text, flags=re.MULTILINE))
    return df

titleDataText =  titleData.loc[:,["text"]]

titleData.loc[:,"text" ] =   prepareTitle(titleDataText)


#####
import nltk
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

teststr = commentData.loc[812130,"text"]

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
    
df =  commentData.loc[:,["text"]]
commentData.loc[:,"text" ] =  df.applymap(lambda text: tokenizeAll(text, tokenizer)) 

df =  postData.loc[:,["text"]]
postData.loc[:,"text" ] =  df.applymap(lambda text: tokenizeAll(text, tokenizer)) 

df =  titleData.loc[:,["text"]]
titleData.loc[:,"text" ] =  df.applymap(lambda text: tokenizeAll(text, tokenizer)) 

newDF = pd.concat([commentData, postData, titleData])

newDF = newDF.loc[ newDF["text"].str.len()  > 20, ['text','category']]
newDF.columns
newDF = newDF.astype({"post_id": object, "parent_id": object, "comment_id": object, "text":str, "category" : object})

import csv
newDF.to_csv('JSCH1.csv', index=False,  quoting = csv.QUOTE_NONNUMERIC)

testdf = pd.read_csv('JSCH1.csv')

testdf.head()


