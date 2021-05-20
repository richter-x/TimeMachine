# -*- coding: utf-8 -*-
"""
RichterX TimeMachine API functions

Spyder 5.0.1 
Python 3.7.9
#@author: Y.Kamer 20210516
"""
import datetime as dt
from datetime import timedelta
import numpy as np
import requests
import time
import sys 

def gen_predList(NUM_pred, region='global', yearStart=1990, yearEnd=2020):
    """
    Generates random predictions around the world 
    Duration: 1 to 30 days, radius: 30 to 300 km
    INPUTS
        NUM_pred:   number of predictions
        region:     name of the bounding box region
        yearStart:  starting year of the random predictions
        yearEnd:    ending year of the random predictions
    OUTPUTS
        predList:  prediction dictionary      
    """
    
    #Bounding boxes for some pre-defined regions
    bbox = {
        'test':   {'lat': [1, 1], 'lon': [1, 1]},
        'global':   {'lat': [-90, 90], 'lon': [-180, 180]},
        'chile':    {'lat': [-56.72, -17.49], 'lon': [-109.67, -66.07]},
        'greece':   {'lat': [34.70, 41.74], 'lon': [19.24, 29.72]},
        'italy':    {'lat': [35.28, 47.09], 'lon': [6.62, 18.78]},
        'japan':    {'lat': [20.21, 45.71], 'lon': [122.71, 154.20]},
        'turkey':   {'lat': [35.80, 42.29], 'lon': [25.62, 44.81]},
        };
    np.random.seed(0);
    
    T0  = datenum(dt.datetime(yearStart, 1, 1)); 
    T1  = datenum(dt.datetime(yearEnd, 1, 1));
    
    lim_dur = [1,30];
    lim_lat = bbox[region]['lat'];
    lim_lon = bbox[region]['lon'];
    lim_rad = [30,300];
    
    #Assign random prediction parameters
    tS  = uni_rand([T0,T1],NUM_pred); #start time
    dur = uni_rand(lim_dur,NUM_pred).round(); #duration
    lt  = uni_rand(lim_lat,NUM_pred); #latitude 
    ln  = uni_rand(lim_lon,NUM_pred); #longitude 
    r   = uni_rand(lim_rad,NUM_pred); #radius
    m   = np.ones(NUM_pred)*5;        #lower magnitude limit
    p   = np.ones(NUM_pred)*(-1); #probability
    hit = np.ones(NUM_pred)*-1; #hit:1, miss:0, undef:-1
    
    predList = {
        "tS"    : tS,
        "dur"   : dur,
        "lt"    : lt,
        "ln"    : ln,
        "r"     : r,
        "m"     : m,
        "p"     : p,
        "hit"   : hit
    }
    
    fun_nm = sys._getframe().f_code.co_name;
    print('[',fun_nm,'] => Generated', len(predList['m']), 'predictions');
    return predList;

    
def read_eqCat(fname='comCat_1990_20200325_M5.csv'):
    #Read a CSV earthquake catalog with the following format
    #event time(datenum), lat, lon, depth, magnitude
    
    data = np.loadtxt(fname,delimiter=',');
    eqCat = {
        "time"  : data[:,0],
        "lat"   : data[:,1],
        "lon"   : data[:,2],
        "depth" : data[:,3],
        "mag"   : data[:,4]
    }
    
    fun_nm = sys._getframe().f_code.co_name;
    print('[',fun_nm,'] => Loaded', len(eqCat['mag']), 'events');
    return eqCat;


def test_pred(predList, eqCat):
    #Tests predictions in predList against the earthquake catalog in eqCat
    
    #Get time range of predictions
    predList['tE'] = predList['tS'] + predList['dur'];
    T0 = min(predList['tS']);
    T1 = max(predList['tE']);
    
    #Check only earthquakes within time range
    eqIdx   = np.where(np.logical_and(eqCat['time']>T0, eqCat['time']<T1))[0];
    
    fun_nm = sys._getframe().f_code.co_name;
    print('[',fun_nm,'] => Testing predictions..')
    
    NUM_pred    = predList['hit'].size;
    for p in range(NUM_pred):
        tmpEqTF   = np.logical_and(eqCat['time'][eqIdx]>predList['tS'][p], 
                                   eqCat['time'][eqIdx]<predList['tE'][p]);
        if(sum(tmpEqTF)):
            dist = gcDist(predList['lt'][p], predList['ln'][p], 
                          eqCat['lat'][eqIdx[tmpEqTF]], eqCat['lon'][eqIdx[tmpEqTF]]);
            predList['hit'][p] = any(dist<predList['r'][p]);
        #Show progress bar
        prog_bar((p+1)/NUM_pred);
    
    print('[',fun_nm,'] => Checked', len(predList['hit']), 'predictions,',
          sum(predList['hit']==1), 'hits', sum(predList['hit']==0), 'misses');
    return predList;


def calc_skill(predList):
    '''
    Calculate predictive skill using the prediction probabilities and outcomes
    Takes into account overlapping predictions by selective sampling
     INPUTS
        predList:   prediction dictionary, e.g. output of 'gen_pred'
    OUTPUTS
        skill:      distionary summarizing the predictive skill
        
    '''
    np.random.seed(0); #seed for reproducible output
    
    NUM_rsmp    = 500; #number of resamples to account for overlapping
    NUM_monte   = 500; #number of draws to calculate significance 
    
    predList['tE'] = predList['tS'] + predList['dur'];
    
    #Check which predictions are overlapping
    NUM_pred    = len(predList['hit']);
    connMat     = np.zeros((NUM_pred,NUM_pred)); 
    for i in range(NUM_pred):
        for j in range(NUM_pred):
            if(i > j):
                #Time overlapping?
                if( max(predList['tS'][i],predList['tS'][j]) < 
                    min(predList['tE'][i],predList['tE'][j]) ):
                    distIJ = gcDist(predList['lt'][i], predList['ln'][i], 
                                  predList['lt'][j], predList['ln'][j]);
                    if(distIJ < (predList['r'][i] + predList['r'][j])):
                        connMat[i,j] = 1;
                        connMat[j,i] = 1;
            elif(i==j):
                connMat[j,i] = 1;
                
    #Generate NUM_rsmp independent sets
    indepSets = [[None]  for _ in range(NUM_rsmp)];
    for i in range(NUM_rsmp):
        #randomly pick a prediction
        selID   = []
        pMark   = np.zeros(NUM_pred);
        while all(pMark)==0:
            #mark all overlapping prediction to avoid selecting
            selID.append(int(np.random.choice(np.where(pMark==0)[0])));
            pMark[np.where(connMat[selID[-1],:]==1)[0]] = 1
        indepSets[i] = selID;
    
    #Make several draws with the prediction probabilites
    vecIR   = np.zeros(NUM_rsmp); #information ratio
    vecSgn  = np.zeros(NUM_rsmp); #significance
    vecAPP  = np.zeros(NUM_rsmp); #avg prediction probability
    vecHR   = np.zeros(NUM_rsmp); #hit rate
    
    fun_nm = sys._getframe().f_code.co_name;
    print('[',fun_nm,'] => Sample prediction probabilities');
    
    for i in range(NUM_rsmp):
        predProb    = predList['p'][indepSets[i]];      #prediction probabilities
        predNumHit  = sum(predList['hit'][indepSets[i]]);    #number of True predictions
            
        matRnd      = np.random.rand(NUM_monte,len(predProb));
        matProb     = np.tile(predProb,(NUM_monte,1));
        pVal        = ((matRnd <= matProb).sum(1) >= predNumHit).sum(0)/NUM_monte;
        
        expHR       = np.mean(predProb); #expected hit rate
        obsHR       = predNumHit/len(predProb); #observed hit rate
        vecIR[i]    = obsHR/expHR; 
        vecSgn[i]   = 1-pVal;
        vecAPP[i]   = expHR;
        vecHR[i]    = obsHR; 
        
        prog_bar((i+1)/NUM_rsmp);
    
    skill = {
        "IR"    : round(vecIR.mean(),3),
        "sgn"   : round(100*vecSgn.mean(),1),
        "APP"   : round(vecAPP.mean(),3),
        "HR"    : round(vecHR.mean(),3),
    }
    
    
    print('[',fun_nm,'] => Finished skill calculation');
    
    print('Predictions: \t' + str(NUM_pred));
    print('Avg. Prob.: \t\t' + str(skill['APP']));
    print('Hit rate: \t\t' + str(skill['HR']));
    print('Info. Ratio: \t' + str(skill['IR']));
    print('Significance: \t' + str(skill['sgn']) + '%');
    
    return skill;


def get_RxTM_prob(predList, API_TOKEN, API_LIMIT):
    """
    Assigns RichterX TimeMachine probabilities to a prediction list
    INPUTS
        predList:   prediction dictionary, e.g. output of 'gen_pred'
        API_TOKEN:  secret token for accessing RxTM API
        API_LIMIT:  request limit per minute, set by RxTM
    OUTPUTS
        predList:  modified prediction dictionary 
    
    NOTE: Predictions will be modified to to match the precalculated forecast
    starting date:  floored to daily midnight
    magnitude:      rounded to M5 or M6
    duration:       rounded to 7, 15 or 30 days
    radius:         rounded to 75, 150 or 300 kms
    lat,lon:        gridded @40km^2 resolution
    """
    
    API_URL     = 'https://www.richterx.com/api/rest.php/1/tmachine';
    #API_TOKEN   = 'testToken'; #replace with your own token
    #API_LIMIT   = 60;#request limit per minute
    
    REQ_headers = {'content-type': 'application/json'};
    NUM_pred    = len(predList['p']);
    
    predList['tS_YMD'] = np.zeros(NUM_pred);
    
    fun_nm = sys._getframe().f_code.co_name;
    print('[',fun_nm,'] => Requesting RichterX probabilities...');
    for i in range(NUM_pred):
        #Get RxTM response
        REQ_data = {'token':     API_TOKEN,
                   'lat':       predList['lt'][i],
                   'lon':       predList['ln'][i],
                   'date':      datenum2dt(predList['tS'][i]).strftime("%Y%m%d"),
                   'dur':       predList['dur'][i],
                   'rad':       predList['r'][i],
                   'mag':       predList['m'][i],
                   };
        respo   = requests.post(API_URL, params=REQ_data, headers=REQ_headers);
        respo   = respo.json();
        
        #Modify prediction to match the API precalculations
        if(respo['success']):
            predList['p'][i]   = respo['prob'];  
            predList['lt'][i]  = respo['lat'];  
            predList['ln'][i]  = respo['lon'];
            predList['tS'][i]  = respo['dateDN'];
            predList['dur'][i] = respo['dur'];
            predList['r'][i]   = respo['rad'];
            predList['m'][i]   = respo['mag'];
            predList['tS_YMD'][i]  = respo['dateYMD'];
            
            #Print progress bar
            prcComp = (i+1)/NUM_pred; #percent complete
            prog_bar(prcComp);
            time.sleep(60/API_LIMIT); #let the API server rest
        else:
            print('API request failed');    
            print(respo);
            return;
    return predList;

def gcDist(lat1, lon1, lat2, lon2):
    #Calculates great circle distance using the haversine formula
    
    DG2RAD = np.pi/180;
    lat1 = lat1*DG2RAD; lon1 = lon1*DG2RAD; lat2 = lat2*DG2RAD; lon2 = lon2*DG2RAD;
    a   = (np.sin((lat2-lat1)/2)**2) + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2)**2;
    rng = 6371 * 2 * np.arctan2(np.sqrt(a),np.sqrt(1 - a));
    return rng;

def uni_rand(lim,num):
    #Generates 'num' uniform random numbers within the range 'lim'
    
    return  lim[0] + np.random.rand(num)*(lim[1]-lim[0]);


def datenum(d):
    #Converts datetime to linear datenum format (days since 0 January 0000)
    
    return 366 + d.toordinal() + (d - dt.datetime.fromordinal(d.toordinal())).total_seconds()/(24*60*60)


def datenum2dt(matlab_datenum):
    #Converts datenum format to datetime format
    
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = timedelta(days=matlab_datenum%1) - timedelta(days = 366)
    return day + dayfrac


def prog_bar(prcComp):
    #Simple inline progress bar showing completion percentage
    
    print("\r[%-25s] %d%%" % ('|'*round(prcComp*25), round(prcComp*100)), end='')
    if prcComp==1:
        print('');
