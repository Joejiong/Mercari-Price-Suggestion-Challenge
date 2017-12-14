
import gc
import time
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 10000


def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    pop_category = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    pop_category = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['general_cat'].isin(pop_category), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


def main():
    start_time = time.time()

    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)

    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])
    submission: pd.DataFrame = test[['test_id']]

    del train
    del test
    gc.collect()

    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
    zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = cv.fit_transform(merge['name'])
    print('[{}] Count vectorize `name` completed.'.format(time.time() - start_time))

    cv = CountVectorizer()
    X_category1 = cv.fit_transform(merge['general_cat'])
    X_category2 = cv.fit_transform(merge['subcat_1'])
    X_category3 = cv.fit_transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])
    print('[{}] TFIDF vectorize `item_description` completed.'.format(time.time() - start_time))

    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['brand_name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))

    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name)).tocsr()
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]

    model = Ridge(solver="sag", fit_intercept=True, random_state=205)
    model.fit(X, y)
    print('[{}] Train ridge completed'.format(time.time() - start_time))
    predsR = model.predict(X=X_test)
    print('[{}] Predict ridge completed'.format(time.time() - start_time))

    model = SGDRegressor(alpha=0.000001, penalty='l2', l1_ratio=0.15,  learning_rate='invscaling',
      loss='squared_loss', power_t=0.25, random_state=None, shuffle=True, tol=None, verbose=0,
      warm_start=False, average=False, epsilon=0.1, eta0=0.01, fit_intercept=True)
    model.fit(X, y).sparsify()
    print('[{}] Train sgd completed'.format(time.time() - start_time))
    predsS = model.predict(X=X_test)
    print('[{}] Predict sgd completed.'.format(time.time() - start_time))

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.05, random_state = 144)
    d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
    d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
    watchlist = [d_train, d_valid]

    params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
        'nthread': 4
    }

    params2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 50,
        'verbosity': -1,
        'metric': 'RMSE',
        'nthread': 4
    }

    model = lgb.train(params, train_set=d_train, num_boost_round=10000, valid_sets=watchlist, \
    early_stopping_rounds=1000, verbose_eval=1000)
    predsL = model.predict(X_test)

    print('[{}] Predict lgb 1 completed.'.format(time.time() - start_time))

    train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 101)
    d_train2 = lgb.Dataset(train_X2, label=train_y2, max_bin=8192)
    d_valid2 = lgb.Dataset(valid_X2, label=valid_y2, max_bin=8192)
    watchlist2 = [d_train2, d_valid2]

    model = lgb.train(params2, train_set=d_train2, num_boost_round=5000, valid_sets=watchlist2, \
    early_stopping_rounds=50, verbose_eval=500)
    predsL2 = model.predict(X_test)

    print('[{}] Predict lgb 2 completed.'.format(time.time() - start_time))

    preds = predsR*0.35 + predsS*0.1+ predsL*0.45 + predsL2*0.1

    submission['price'] = np.expm1(preds)
    submission.to_csv("submission_ridge_sgd_2xlgbm.csv", index=False)

if __name__ == '__main__':
    main()