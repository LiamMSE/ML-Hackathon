# 
# Analytics Vidhya - hackathon
# This is personal project inspired by 5th place solution - varunbpatil @ Dec 2018
#
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Read training and test datasets.
df_train = pd.read_csv('train_GzS76OK/train.csv')
df_center_info = pd.read_csv('train_GzS76OK/fulfilment_center_info.csv')
df_meal_info = pd.read_csv('train_GzS76OK/meal_info.csv')
df_test = pd.read_csv('test_QoiMO9B.csv')


# Merge the training data with the branch and meal information by left joinging them
df_train = pd.merge(df_train, df_center_info, how="left", left_on='center_id', right_on='center_id')

df_train = pd.merge(df_train, df_meal_info, how='left', left_on='meal_id', right_on='meal_id')


# Merge the test data with the branch and meal information.
df_test = pd.merge(df_test, df_center_info, how="left", left_on='center_id', right_on='center_id')

df_test = pd.merge(df_test, df_meal_info, how='left', left_on='meal_id', right_on='meal_id')


# Convert 'city_code' and 'region_code' into a single feature - 'city_region'.
df_train['city_region'] = df_train['city_code'].astype('str') + '_' + \
        df_train['region_code'].astype('str')

df_test['city_region'] = df_test['city_code'].astype('str') + '_' + \
        df_test['region_code'].astype('str')


# Label encode categorical columns for use in LightGBM.
label_encode_columns = ['center_id',
                        'meal_id',
                        'city_code',
                        'region_code',
                        'city_region',
                        'center_type',
                        'category',
                        'cuisine']

le = preprocessing.LabelEncoder()

for col in label_encode_columns:
    le.fit(df_train[col])
    df_train[col + '_encoded'] = le.transform(df_train[col])
    df_test[col + '_encoded'] = le.transform(df_test[col])


# Feature engineering - treat 'week' as a cyclic feature.
# Encode it using sine and cosine transform.
df_train['week_sin'] = np.sin(2 * np.pi * df_train['week'] / 52.143)

df_train['week_cos'] = np.cos(2 * np.pi * df_train['week'] / 52.143)

df_test['week_sin'] = np.sin(2 * np.pi * df_test['week'] / 52.143)

df_test['week_cos'] = np.cos(2 * np.pi * df_test['week'] / 52.143)


# Feature engineering - percent difference between base price and checkout price.
df_train['price_diff_percent'] = \
        (df_train['base_price'] - df_train['checkout_price']) / \
        df_train['base_price']

df_test['price_diff_percent'] = \
        (df_test['base_price'] - df_test['checkout_price']) / \
        df_test['base_price']


# Convert email and homepage features into a single feature - 'email_plus_homepage'.
df_train['email_plus_homepage'] = \
        df_train['emailer_for_promotion'] + \
        df_train['homepage_featured']

df_test['email_plus_homepage'] = \
        df_test['emailer_for_promotion'] + \
        df_test['homepage_featured']


# Prepare a list of columns to train on.
# Also decide which features to treat as numeric and which features to treat
# as categorical.
columns_to_train = ['week',
                    'week_sin',
                    'week_cos',
                    'checkout_price',
                    'base_price',
                    'price_diff_percent',
                    'email_plus_homepage',
                    'city_region_encoded',
                    'center_type_encoded',
                    'op_area',
                    'category_encoded',
                    'cuisine_encoded',
                    'center_id_encoded',
                    'meal_id_encoded']

categorical_columns = ['email_plus_homepage',
                       'city_region_encoded',
                       'center_type_encoded',
                       'category_encoded',
                       'cuisine_encoded',
                       'center_id_encoded',
                       'meal_id_encoded']

numerical_columns = [col for col in columns_to_train if col not in categorical_columns]


# Log transform the target variable - num_orders.
df_train['num_orders_log1p'] = np.log1p(df_train['num_orders'])


# Train-Test split.
X = df_train[categorical_columns + numerical_columns]
y = df_train['num_orders_log1p']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.02,shuffle=False)

scores = []
params = []

param_grid_choices = {'num_leaves': [128, 255, 512, 1024],
              'min_child_samples': [5, 10, 15, 30],
              'colsample_bytree': [0.2, 0.3, 0.4, 0,5, 0.6, 0.7]}

estimator = LGBMRegressor(learning_rate=0.003, n_estimators=1000, silent=True, **param_grid_choices)

gs = GridSearchCV(estimator,param_grid=param_grid_choices, n_jobs=-1)

fit_params = {'early_stopping_rounds': 1000, 'feature_name': categorical_columns + numerical_columns,
              'categorical_feature': categorical_columns,
              'eval_set': [(X_test, y_test)]}
#estimator.fit(X_train, y_train, **fit_params)
fitted_model = gs.fit(X_train, y_train, **fit_params)
print("Best score = {}".format(fitted_model.best_score_))
print("Best params = {}".format(fitted_model.best_params_))

# for i, g in enumerate(ParameterGrid(param_grid_choices)):
#     print("param grid: {}/{}".format(i, len(ParameterGrid(param_grid_choices)) - 1))
#     estimator = LGBMRegressor(learning_rate=0.003, n_estimators=1000, silent=True, **g)
#     fit_params = {'early_stopping_rounds': 1000, 'feature_name': categorical_columns + numerical_columns,
#                   'categorical_feature': categorical_columns,
#                   'eval_set': [(X_test, y_test)]}
#     estimator.fit(X_train, y_train, **fit_params)
#
#     print("estimator.best_score = {} with this param {}".format(estimator.best_score_, g))
#     scores.append(estimator.best_score_['l2'])
#     #scores.append(estimator.best_score_['l2'])
#     params.append(g)
#
# scores_arry = np.array(scores)
# print(scores_arry[0])
# print(scores_arry[1])
# type(scores_arry)
# print("Best score = {}".format(np.min(scores)))
#
# print("Best params = {}".format(params[np.argmin(scores_arry)]))
# #print(params[np.argmin(scores['l2]'])])

# Train the LightGBM model on the best parameters obtained by grid search.
# g = {'colsample_bytree': 0.4,
#      'min_child_samples': 5,
#      'num_leaves': 255,
#      'max_depth': 8,
#      'feature_fraction': 0.6}
#
# estimator = LGBMRegressor(learning_rate=0.003,
#                           n_estimators=5000,
#                           silent=True,
#                           **g)
#
# fit_params = {'early_stopping_rounds': 1000,
#               'feature_name': categorical_columns + numerical_columns,
#               'categorical_feature': categorical_columns,
#               'eval_set': [(X_train, y_train), (X_test, y_test)]}
#
# estimator.fit(X_train, y_train, **fit_params)
#
# scores.append(estimator.best_score_['valid_1']['l2'])
# params.append(g)
#
# # print("Best score = {}".format(np.min(scores)))
# # print("Best params =")
# # print(params[np.argmin(scores)])

#
# # Get predictions on the test set and prepare submission file.
# X = df_test[categorical_columns + numerical_columns]
#
# pred = estimator.predict(X)
# pred = np.expm1(pred)
#
# submission_df = df_test.copy()
# submission_df['num_orders'] = pred
# submission_df = submission_df[['id', 'num_orders']]
# submission_df.to_csv('submission.csv', index=False)
