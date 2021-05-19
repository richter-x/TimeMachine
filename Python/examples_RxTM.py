# -*- coding: utf-8 -*-
"""
RichterX TimeMachine API Examples

#@author: Y.Kamer 20210516
"""
import functions_RxTM as rx;
from functions_RxTM import dt
from functions_RxTM import np

################################################################
# Example 1:
# Two predictions with p=0.16 and p=0.50, one hit one miss
################################################################
print('==== Example 1: ==== ');
predList = {
    "tS"    : np.array([rx.datenum(dt.datetime(2018,1,1)),
                        rx.datenum(dt.datetime(2019,1,1))]),
    "dur"   : np.array([10,     30]),
    "lt"    : np.array([22,     22]),
    "ln"    : np.array([44,     44]),
    "r"     : np.array([100,    50]),
    "m"     : np.array([5,      5]),
    "p"     : np.array([0.16,   0.50]),
    "hit"   : np.array([0,      1])
}
rx.calc_skill(predList);

################################################################
# Example 2:
# 10 random predictions in Italy 
# Assign probability p=0.01 and calculate skill
################################################################
print('==== Example 2: ==== ');
numPred     = 10;
region      = 'italy';
yearStart   = 2000;
yearEnd     = 2010;

#Generate random predictions
predList        = rx.gen_predList(numPred, region, yearStart, yearEnd);
predList['p'][:]    = 0.01;
predList['dur'][:]  = 30;
predList['r'][:]    = 300;

#Load event catalog for testing
eqCat       = rx.read_eqCat();

#Test predictions
predList    = rx.test_pred(predList,eqCat);

#Calculate skill 
rx.calc_skill(predList);

################################################################
# Example 3:
# 10 random predictions in Italy 
# Assign probability from TimeMachine API and calculate skill
################################################################
print('==== Example 3: ==== ');

#Set RxTM API token and limit
RxApiToken  = 'testToken';  #replace with your own token
RxApiLimit  = 60;           #request limit per minute

#Assign probability using the Rx TimeMachine API
#NOTE: Prediction parameters will change to match the pre-calculated forecasts
predList = rx.get_RxTM_prob(predList, RxApiToken, RxApiLimit);

#Test predictions
rx.test_pred(predList, eqCat);

#Calculate skill 
rx.calc_skill(predList);

