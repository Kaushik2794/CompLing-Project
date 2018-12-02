# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:32:26 2018

@author: kaush
"""
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import re

from nltk.corpus import stopwords

import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

##define a function for auto ml

def auto_ML(model, Xtrain, Ytrain, Xtest, Ytest, folds= 10):
    model.fit(Xtrain, Ytrain)
    prediction= model.predict(Xtest)
    
    # Cross validation
    scores= cross_val_score(model, Xtrain, Ytrain, cv= folds)
    val_score= sum(scores/folds)
    
    print('Test prediction:\n{}'.format(prediction))
    print('-------------------------------------')
    print('Accuracy score: {}'.format(accuracy_score(Ytest, prediction)))
    print('-------------------------------------')
    print('confusion matrix:\n {}'.format(confusion_matrix(Ytest, prediction)))
    print('-------------------------------------')
    print('Cross Validation score: {}'.format(val_score))
	
train= pd.read_csv('/Users/akshatpant/Desktop/UMD/Sem 3/Comp Ling/Project/train.csv')
test= pd.read_csv('/Users/akshatpant/Desktop/UMD/Sem 3/Comp Ling/Project/test.csv')

"""
Add new feature: Question length

Add new features: Extract max IR score and correscponding page from IR_Wiki Scores
Also calculate difference between highest and second highest IR score

Add target feature paren_match (true if wiki page and answer match)
Apply Weight of Evidence(WOE) encoding to category variable
"""

#Question length (training data)
lens= []
for i in range(0, train.shape[0]):
    lens.append(len(train.loc[i]['Question Text']))
lens
train['Quest len']= lens

#Question length (testing data)
lens= []
for i in range(0, test.shape[0]):
    lens.append(len(test.loc[i]['Question Text']))
lens
test['Quest len']= lens

######################################################################

# Max IR score and corresponding page (training data)
wiki_page= []
page_score= []
diff= []
for i in range(0, train.shape[0]):
    ans_score= {}
    for ii in train.loc[i]['IR_Wiki Scores'].split(', '):
        ans_score[ii.split(':')[0]]= float(ii.split(':')[1])
        
    
    page= sorted(ans_score, key= ans_score.get, reverse= True)[0]
    page2= sorted(ans_score, key= ans_score.get, reverse= True)[1]
    wiki_page.append(page)
    page_score.append(ans_score[page])
    diff.append(ans_score[page]- ans_score[page2])

train['Wiki page']= wiki_page
train['Page score']= page_score
train['Score difference']= diff

# Max IR score and corresponding page (testing data)
wiki_page= []
page_score= []
diff= []
for i in range(0, test.shape[0]):
    ans_score= {}
    for ii in test.loc[i]['IR_Wiki Scores'].split(', '):
        ans_score[ii.split(':')[0]]= float(ii.split(':')[1])
        
    
    page= sorted(ans_score, key= ans_score.get, reverse= True)[0]
    page2= sorted(ans_score, key= ans_score.get, reverse= True)[1]
    wiki_page.append(page)
    page_score.append(ans_score[page])
    diff.append(ans_score[page]- ans_score[page2])

test['Wiki page']= wiki_page
test['Page score']= page_score
test['Score difference']= diff

################################################################

# Target feature paren_match (training data). It is 1 if answer and wiki page match. 0 otherwise
train['paren_match']= 0

for i, row in train.iterrows():
    if row['Answer'] == row['Wiki page']:
        train.loc[i, 'paren_match']= 1
        


#################################################################

# WOE encoding

encoding= ce.WOEEncoder(cols= ['category', 'Wiki page'])
encoding.fit(train, train['paren_match'])
train_df=encoding.transform(train)

features= ['Wiki page', 'Quest len', 'Page score',  'category', 'Score difference']
target= ['paren_match']

scaler= StandardScaler()

scaler.fit(train_df[features].values)

train_df[features]= scaler.transform(train_df[features].values)

train_df.head()


X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df[target], test_size=0.20)

y_train= np.reshape(y_train.values, (y_train.shape[0], ))

auto_ML(svm_clf, X_train, y_train, X_test, y_test, folds= 10)


################################################################

# LSTM

"""
prepare vocabulary from text
prepare word_to_index dictionary
set <pad> and <unk> value to 0
normalize text before adding to vocab
remove stopwords
"""

regex= re.compile(r"\b(\w*['\w]*[\w]*)[^\w]*")

vocab= {'<PAD>': 0, '<UNK>': 0}
word_to_ix= {}

for i, row in test.iterrows():
    sent= train.loc[i, 'Question Text'].lower()
    for word in regex.findall(sent):
        if word not in vocab and len(word)> 1 and word not in stopwords.words('english'):
            vocab[word]= len(vocab)- 1
    train.loc[i, 'Question Text']= ' '.join(regex.findall(sent))
    
vocab



"""
Create LSTM network for the text.
Combine other features into the output of LSTM.

"""

class LSTM_special(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, add_dim, out_dim, num_layers= 1):
        super(LSTM_special, self).__init__()
        self.input_dim= input_dim
        self.hidden_dim= hidden_dim
        self.batch_size= batch_size
        self.add_dim= add_dim
        self.out_dim= out_dim
        self.num_layers= num_layers
        
        # Define the LSTM layer
        self.lstm= nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        
        # Define the linear layer that maps from (hidden_dim + add_dim) to out_dim
        self.linear= nn.Linear(self.hidden_dim+self.add_dim, self.out_dim)
        
        # Define the non-linearity that converts to probability
        self.softmax= nn.Softmax()
        
    def init_hidden(self):
        """
        Initialize the hidden state (h0, c0)
        
        Before we've done anything, we dont have any hidden state.
        Refer to the Pytorch documentation to see exactly
        why they have this dimensionality.
        The axes semantics are (num_layers, minibatch_size, hidden_dim)
        """
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
               torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        
    def forward(self, sequence, add_features):
        """
        forward pass through LSTM layer
        hidden to output space mapping by Linear layer
        lstm_out shape: [seq_len/input_len, batch_size, hidden_dim]
        self.hidden shape= (a, b) where a & b both have shape: [num_layers, batch_size, hidden_dim]
        """
        lstm_out, self.hidden= self.lstm(sequence.view(seq_len, self.batch_size, -1), self.hidden)
        
        """
        Take the output from the last layer of the LSTM and 
        concatenate the additional features to them.
        Map them to output space.
        Apply non linearity like softmax
        """
        # get the output from the last timestep
        lstm_out= lstm_out[-1].view(self.batch_size, -1)
        
        # concatenate additional features to lstm output
        new_features= torch.cat((lstm_out, add_features.view(self.batch_size, -1)), 1)
        
        # map to output space
        y_pred= self.linear(new_features)
        
        # apply non linearity
        output= self.softmax(y_pred)
        
        
        return output