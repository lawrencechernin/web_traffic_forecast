import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import gc; gc.enable()
from sklearn.feature_extraction import text
from sklearn import naive_bayes
import matplotlib.pyplot as plt

full_data = pd.read_csv("../input/train_2.csv") # remove 'nrows' to test all data
full_data = full_data.fillna(0.)

# Cross validation will be run for the following lookbacks
# After all CVs are run a plot of 5 fold avergae SMAPE score vs look_back is generated

look_backs = [15, 18, 20, 23, 25, 27, 30, 33, 35]
SAMPLE = True # Set false to use entire data set
SAMPLES = 8000
#look_backs = [25]

# Randomly sample rows if specified
if SAMPLE:
    full_data = full_data.sample(SAMPLES, axis=0).reset_index()
    full_data.index = range(len(full_data))
    print("Using", SAMPLES, "random rows")

def main():
    smape_avgs = []
    for lb in look_backs:
        smape_avgs.append(runCV(lb))

    plt.plot(look_backs, smape_avgs)
    plt.scatter(look_backs, smape_avgs)
    plt.xlabel("look_back")
    plt.ylabel("5 Fold SMAPE")
    plt.show()

def runCV(look_back):
    #build cross validation sets, make predictions, and calculate SMAPES
    k = 5
    gap = 61 # Cross Validation gap
    predictionDays = 61 # Number of days to predict
    smapes = []


    for i in range(k):
        # Determine last day to predict

        end = (-1*gap * (i + 1) + predictionDays)
        # Divide into training and test sets
        train = full_data.iloc[:, 0:(-1*gap * (i + 1))]
        test = full_data[['Page'] + list(full_data.columns[(-1*gap * (i + 1)):(-1*gap * (i + 1) + predictionDays)])]
        if end >= 0:
            test = full_data[['Page'] + list(full_data.columns[(-1*gap * (i + 1)):])]
        else:
            test = full_data[['Page'] + list(full_data.columns[(-1*gap * (i + 1)):end])]

        # melted set is easier to work with and closier to submission format
        test_melt = pd.melt(test, id_vars='Page', var_name='date', value_name='Visits')
        print("Training from", train.columns.values[1], "to", train.columns.values[-1])
        print("Testing from", test.columns.values[1], "to", test.columns.values[-1])
        smapes.append(makePredictions(train, test_melt, look_back))

    print("Average SMAPES:", np.mean(smapes))
    print("\n")
    return np.mean(smapes)

def smape(y_true, y_pred):
    # this function is used to compute smape scores
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return np.nanmean(diff)

# I'm gong to share a solution that I found interesting with you.
# The idea is to compute the median of the series in different window sizes at the end of the series,
# and the window sizes are increasing exponentially with the base of golden ratio.
# Then a median of these medians is taken as the estimate for the next 60 days.
# This code's result has the score of around 44.9 on public leaderboard, but I could get upto 44.7 by playing with it.

def makePredictions(train, test_melt, look_back=49):
    r = 1.61803398875
    Windows = np.round(r**np.arange(1,9) * 7)
    Windows = [11, 18, 30, 48, 78, 126, 203, 329]

    n = train.shape[1] - 1 #  550
    Visits = np.zeros(train.shape[0])
    for i, row in train.iterrows():
        M = []
        start = row[1:].nonzero()[0]
        if len(start) == 0:
            continue
        if n - start[0] < Windows[0]:
            Visits[i] = row.iloc[start[0]+1:].median()
            continue
        for W in Windows:
            if W > n-start[0]:
                break
            M.append(row.iloc[-W:].median())
        Visits[i] = np.median(M)

    Visits[np.where(Visits < 1)] = 0.
    
    train['Predicted'] = Visits
    #print(train.head())
    #test1 = pd.read_csv("../input/key_2.csv")
    #test1['Page'] = test1.Page.apply(lambda x: x[:-11])
    test1 = test_melt.merge(train[['Page','Predicted']], on='Page', how='left')
    #print('MODEL 1 SMAPE: ', smape(test1['Visits'], test1['Predicted']))


    # add model 2
    

    #determine idiom with URL
    train['origine']=train['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])

    '''
    This is what you get with a value counts on train.origine
    en    24108
    ja    20431
    de    18547
    fr    17802
    zh    17229
    ru    15022
    es    14069
    ts    13556
    er     4299
    '''
    #we have english, japanese, deutch, french, chinese (taiwanese ?), russian, spanish
    #ts and er are undetermined; in the next lines, I try to replace them by learning from special chars
    #Note : this step wasn't tuned, and can't be perfect because other idioms are available in those Pages (such as portuguese for example)

    #let's make a train, target, and test to predict language on ts and er pages
    orig_train=train.loc[~train.origine.isin(['ts', 'er']), 'Page']
    orig_target=train.loc[~train.origine.isin(['ts', 'er']), 'origine']
    orig_test=train.loc[train.origine.isin(['ts', 'er']), 'Page']
    #keep only interesting chars
    orig_train2=orig_train.apply(lambda x:x.split(".wikipedia")[0][:-3]).apply(lambda x:re.sub("[a-zA-Z0-9():\-_ \'\.\/]", "", x))
    orig_test2=orig_test.apply(lambda x:x.split(".wikipedia")[0][:-3]).apply(lambda x:re.sub("[a-zA-Z0-9():\-_ \'\.\/]", "", x))
    #run TFIDF on those specific chars
    tser_model = True
    try:
        tfidf=text.TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, 
                            lowercase=True, preprocessor=None, tokenizer=None, 
                            analyzer='char', #stop_words=[chr(x) for x in range(97,123)]+[chr(x) for x in range(65,91)]+['_','.',':'], 
                            token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=True, norm='l2', 
                            use_idf=True, smooth_idf=True, sublinear_tf=False)
        orig_train2=tfidf.fit_transform(orig_train2)
        #apply a simple naive bayes on the text features
        model=naive_bayes.BernoulliNB()
        model.fit(orig_train2, orig_target)
        result=model.predict(tfidf.transform(orig_test2))
        result=pd.DataFrame(result, index=orig_test)
        result.columns=['origine']
    except:
        tser_model = False
    #result will be used later to replace 'ts' and 'er' values
    #we need to remove train.origine so that the train can be flattened with melt
    del train['origine']
    del train['Predicted']

    #let's flatten the train as did clustifier and initialize a "ferie" columns instead of a weekend column
    #look_back=49
    #look_back=50
    #look_back=51
    #look_back=54
    #look_back=60

    train = pd.melt(train[list(train.columns[-look_back:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
    train['date'] = train['date'].astype('datetime64[ns]')
    train['ferie'] = ((train.date.dt.dayofweek) >=5).astype(float)
    train['origine']=train['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])

    #let's join with result to replace 'ts' and 'er'
    if tser_model:
        join=train.loc[train.origine.isin(["ts","er"]), ['Page']]
        join['origine']=0 #init
        join.index=join["Page"]
        join.origine=result
        train.loc[train.origine.isin(["ts","er"]), ['origine']]=join.origine.values #replace

    #official non working days by country (manual search with google)
    #I made a lot of shortcuts considering that only Us and Uk used english idiom, 
    #only Spain for spanich, only France for french, etc
    train_us=['2015-07-04','2015-11-26','2015-12-25']+\
    ['2016-07-04','2016-11-24','2016-12-26']
    test_us=[]
    train_uk=['2015-12-25','2015-12-28'] +\
    ['2016-01-01','2016-03-28','2016-05-02','2016-05-30','2016-12-26','2016-12-27']
    test_uk=['2017-01-01']
    train_de=['2015-10-03', '2015-12-25', '2015-12-26']+\
    ['2016-01-01', '2016-03-25', '2016-03-26', '2016-03-27', '2016-01-01', '2016-05-05', '2016-05-15', '2016-05-16', '2016-10-03', '2016-12-25', '2016-12-26']
    test_de=['2017-01-01']
    train_fr=['2015-07-14', '2015-08-15', '2015-11-01', '2015-11-11', '2015-12-25']+\
    ['2016-01-01','2016-03-28', '2016-05-01', '2016-05-05', '2016-05-08', '2016-05-16', '2016-07-14', '2016-08-15', '2016-11-01','2016-11-11', '2016-12-25']
    test_fr=['2017-01-01']
    train_ru=['2015-11-04']+\
    ['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-01-05', '2016-01-06', '2016-01-07', '2016-02-23', '2016-03-08', '2016-05-01', '2016-05-09', '2016-06-12', '2016-11-04']
    test_ru=['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', '2017-02-23']
    train_es=['2015-08-15', '2015-10-12', '2015-11-01', '2015-12-06', '2015-12-08', '2015-12-25']+\
    ['2016-01-01', '2016-01-06', '2016-03-25', '2016-05-01', '2016-08-15', '2016-10-12', '2016-11-01', '2016-12-06', '2016-12-08', '2016-12-25']
    test_es=['2017-01-01', '2017-01-06']
    train_ja=['2015-07-20','2015-09-21', '2015-10-12', '2015-11-03', '2015-11-23', '2015-12-23']+\
    ['2016-01-01', '2016-01-11', '2016-02-11', '2016-03-20', '2016-04-29', '2016-05-03', '2016-05-04', '2016-05-05', '2016-07-18', '2016-08-11', '2016-09-22', '2016-10-10', '2016-11-03', '2016-11-23', '2016-12-23']
    test_ja=['2017-01-01', '2017-01-09', '2017-02-11']
    train_zh=['2015-09-27', '2015-10-01', '2015-10-02','2015-10-03','2015-10-04','2015-10-05','2015-10-06','2015-10-07']+\
    ['2016-01-01', '2016-01-02', '2016-01-03', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-04-04', '2016-05-01', '2016-05-02', '2016-06-09', '2016-06-10', '2016-09-15', '2016-09-16', '2016-10-03', '2016-10-04','2016-10-05','2016-10-06','2016-10-07']
    test_zh=['2017-01-02', '2017-02-27', '2017-02-28', '2017-03-01']
    #in China some saturday and sundays are worked
    train_o_zh=['2015-10-10','2016-02-06', '2016-02-14', '2016-06-12', '2016-09-18', '2016-10-08', '2016-10-09']
    test_o_zh=['2017-01-22', '2017-02-04']

    #let's replace values in 'ferie' columns
    train.loc[(train.origine=='en')&(train.date.isin(train_us+train_uk)), 'ferie']=1
    train.loc[(train.origine=='de')&(train.date.isin(train_de)), 'ferie']=1
    train.loc[(train.origine=='fr')&(train.date.isin(train_fr)), 'ferie']=1
    train.loc[(train.origine=='ru')&(train.date.isin(train_ru)), 'ferie']=1
    train.loc[(train.origine=='es')&(train.date.isin(train_es)), 'ferie']=1
    train.loc[(train.origine=='ja')&(train.date.isin(train_ja)), 'ferie']=1
    train.loc[(train.origine=='zh')&(train.date.isin(train_zh)), 'ferie']=1
    train.loc[(train.origine=='zh')&(train.date.isin(train_o_zh)), 'ferie']=0

    #same with test
    #test = pd.read_csv("../input/key_2.csv")
    test = test_melt
    del test['Visits']
    #test['date'] = test.Page.apply(lambda a: a[-10:])
    #test['Page'] = test.Page.apply(lambda a: a[:-11])
    test['date'] = test['date'].astype('datetime64[ns]')
    test['ferie'] = ((test.date.dt.dayofweek) >=5).astype(float)
    test['origine']=test['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])

    #joint with result
    if tser_model:
        join=test.loc[test.origine.isin(["ts","er"]), ['Page']]
        join['origine']=0
        join.index=join["Page"]
        join.origine=result
        test.loc[test.origine.isin(["ts","er"]), ['origine']]=join.origine.values

    test.loc[(test.origine=='en')&(test.date.isin(test_us+test_uk)), 'ferie']=1
    test.loc[(test.origine=='de')&(test.date.isin(test_de)), 'ferie']=1
    test.loc[(test.origine=='fr')&(test.date.isin(test_fr)), 'ferie']=1
    test.loc[(test.origine=='ru')&(test.date.isin(test_ru)), 'ferie']=1
    test.loc[(test.origine=='es')&(test.date.isin(test_es)), 'ferie']=1
    test.loc[(test.origine=='ja')&(test.date.isin(test_ja)), 'ferie']=1
    test.loc[(test.origine=='zh')&(test.date.isin(test_zh)), 'ferie']=1
    test.loc[(test.origine=='zh')&(test.date.isin(test_o_zh)), 'ferie']=0

    train_page_per_dow = train.groupby(['Page','ferie']).median().reset_index()
    test = test.merge(train_page_per_dow, on=['Page', 'ferie'], how='left')

    test['Pred2'] = test['Visits']
    test.loc[test.Pred2.isnull(), 'Pred2'] = 0
    test['PredC']=((test['Pred2']*10).astype('int')/10 + test1['Predicted'])/2

    test['Visits'] = test1['Visits']
    test['Pred1'] = test1['Predicted']

    #print("MODEL 2 SMAPE: ", smape(test['Visits'], test['Pred2']))

    combinedSmape = smape(test['Visits'], test['PredC'])
    print("Combined SMAPE: ", combinedSmape)
    print("look_back:", look_back)
    print("------------------------------------")
    #test[['Id','Visits']].to_csv('sub.csv', index=False)
    return(combinedSmape)

# Run code
main()