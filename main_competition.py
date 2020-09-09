
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

import sys
sys.path.insert(0, 'C:/Users/Max Power/OneDrive/ponte/programmi/python/progetto2/AJ_lib')
# sys.path.insert(0, 'C:/Users/ajacassi/OneDrive/ponte/programmi/python/progetto2/AJ_lib')

import json
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

from AJ_models_classifier import learning_class
from preprocess_flu import preprocess_data

import streamlit as st

@st.cache
def load_file(name):
    df = pd.read_csv('data/'+name+'.csv')
    return df

df = load_file('submission_format')
df_sub = df.copy()
df = load_file('test_set_features')
df_test_x = df.copy()
df = load_file('training_set_features')
df_train_x = df.copy()
df = load_file('training_set_labels')
df_train_y = df.copy()

st.write(df_sub)
# st.write(df_test_x)
# st.write(df_train_x)
# st.write(df_train_y)

#########################################################

df_data_x, global_dict = preprocess_data().eda(df_train_x)
df_data = df_data_x.copy()
df_data['seasonal_vaccine'] = df_train_y['seasonal_vaccine']
df_data['h1n1_vaccine'] = df_train_y['h1n1_vaccine']
st.write(df_data)

#########################################################

df_train, df_test = train_test_split(df_data, test_size=0.3, random_state = 222)

X_train = df_train.drop(['respondent_id', 'seasonal_vaccine', 'h1n1_vaccine'], axis = 1)
X_test = df_test.drop(['respondent_id', 'seasonal_vaccine', 'h1n1_vaccine'], axis = 1)

Y1_train = df_train['seasonal_vaccine']
Y1_test = df_test['seasonal_vaccine']

Y2_train = df_train['h1n1_vaccine']
Y2_test = df_test['h1n1_vaccine']

#######################################################

X_sub_train = df_data.drop(['respondent_id', 'seasonal_vaccine', 'h1n1_vaccine'], axis = 1)
Y1_sub_train = df_data['seasonal_vaccine']
Y2_sub_train = df_data['h1n1_vaccine']

df_test_x, global_dict = preprocess_data().eda(df_test_x)
df_submit_x = df_test_x.copy()
df_submit_x = df_submit_x.drop(['respondent_id'], axis = 1)

#######################################################


# ███████ ██ ███    ███ ██████  ██      ███████
# ██      ██ ████  ████ ██   ██ ██      ██
# ███████ ██ ██ ████ ██ ██████  ██      █████
#      ██ ██ ██  ██  ██ ██      ██      ██
# ███████ ██ ██      ██ ██      ███████ ███████

Y_train = Y1_train
Y_test = Y1_test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

learn = learning_class()
models = learn.get_models(['XGBClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier'])

models, history = learn.train_models(models, X_train, Y_train)
models = learn.calibration_model(models, X_train, Y_train)

models['deep learning normal binary'] = learn.get_deep_learning_model(input_dl = X_train.shape[1], net_type = 'normal', loss_type = 'binary')

models, history = learn.train_models(models, X_train, Y_train, epochs = 50)
# validation_data = [(X_train, Y_train),(X_test, Y_test)]
# models, history = learn.train_models(models, X_train, Y_train, validation_data = validation_data, epochs = 10)
# learn.plot_history(history['XGBClassifier'], type='xgb')
# st.pyplot()

y_prob_dict, y_pred_matrix = learn.prob_matrix_generator(models, X_test, 2)
score_data = learn.score_models(models, Y_test, y_prob_dict, y_pred_matrix)
st.write(score_data)
learn.plot_roc_curve(y_prob_dict, Y_test)
st.pyplot()


# ██████  ███    ██ ███    ██
# ██   ██ ████   ██ ████   ██
# ██   ██ ██ ██  ██ ██ ██  ██
# ██   ██ ██  ██ ██ ██  ██ ██
# ██████  ██   ████ ██   ████


# Y_sparse_train = pd.DataFrame()
# Y_sparse_train['h1n1_vaccine'] = Y2_train
# Y_sparse_train['seasonal_vaccine'] = Y1_train
#
# Y_sparse_test = pd.DataFrame()
# Y_sparse_test['h1n1_vaccine'] = Y2_test
# Y_sparse_test['seasonal_vaccine'] = Y1_test
#
# learn = learning_class()
# models = dict()
# net_type = 'normal'
# loss_type = 'multisparse'
# models['deep learning '+net_type+' '+loss_type] = learn.get_deep_learning_model(input_dl = X_train.shape[1], net_type = net_type, loss_type = loss_type, num_classes = 2)
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# # inverse = scaler.inverse_transform(X_train)
#
# models, history = learn.train_models(models, X_train, Y_sparse_train, epochs = 30)
# # validation_data = (X_test, Y_sparse_test)
# # models, history = learn.train_models(models, X_train, Y_sparse_train, validation_data = validation_data, epochs = 100)
# # learn.plot_history(history['deep learning '+net_type+' '+loss_type], type='dnn')
# # st.pyplot()
#
# y_prob_dict, y_pred_matrix = learn.prob_matrix_generator(models, X_test, 2)
# st.write(y_prob_dict)
# st.write(y_pred_matrix)
#
# temp_dict = {'deep learning '+net_type+' '+loss_type: y_prob_dict, 'Ensamble': y_prob_dict}
# temp_df = pd.DataFrame()
# temp_df['deep learning '+net_type+' '+loss_type] = y_pred_matrix[1]
# temp_df['Ensamble'] = y_pred_matrix[1]
# score_data = learn.score_models(models, Y_sparse_test['h1n1_vaccine'], temp_dict, temp_df)
# st.write(score_data)
# learn.plot_roc_curve(temp_dict, Y_sparse_test['h1n1_vaccine'])
# st.pyplot()
#
# temp_dict = {'deep learning '+net_type+' '+loss_type: y_prob_dict, 'Ensamble': y_prob_dict}
# temp_df = pd.DataFrame()
# temp_df['deep learning '+net_type+' '+loss_type] = y_pred_matrix[0]
# temp_df['Ensamble'] = y_pred_matrix[0]
# score_data = learn.score_models(models, Y_sparse_test['seasonal_vaccine'], temp_dict, temp_df)
# st.write(score_data)
# learn.plot_roc_curve(temp_dict, Y_sparse_test['seasonal_vaccine'])
# st.pyplot()

# ██   ██ ██    ██ ██████  ███████ ██████
# ██   ██  ██  ██  ██   ██ ██      ██   ██
# ███████   ████   ██████  █████   ██████
# ██   ██    ██    ██      ██      ██   ██
# ██   ██    ██    ██      ███████ ██   ██


# with open('param_search') as f:
#     grid_search = json.load(f)
#
# learn = learning_class()
# models = learn.get_models(['XGBClassifier'])

# dict_best_param = learn.train_hyperparameters(models, X_train, Y1_train, grid_search, filename = 'best_param1', type = 'grid')
# st.write(dict_best_param)
# dict_best_param = learn.train_hyperparameters(models, X_train, Y2_train, grid_search, filename = 'best_param2', type = 'grid')
# st.write(dict_best_param)
# learning_rate = 0.2,  max_depth = 4, n_estimator = 100


# ███████ ███████  █████  ████████ ██    ██ ██████  ███████     ███████ ███████ ██      ███████  ██████ ████████ ██  ██████  ███    ██
# ██      ██      ██   ██    ██    ██    ██ ██   ██ ██          ██      ██      ██      ██      ██         ██    ██ ██    ██ ████   ██
# █████   █████   ███████    ██    ██    ██ ██████  █████       ███████ █████   ██      █████   ██         ██    ██ ██    ██ ██ ██  ██
# ██      ██      ██   ██    ██    ██    ██ ██   ██ ██               ██ ██      ██      ██      ██         ██    ██ ██    ██ ██  ██ ██
# ██      ███████ ██   ██    ██     ██████  ██   ██ ███████     ███████ ███████ ███████ ███████  ██████    ██    ██  ██████  ██   ████


# from lib_ML_eda import learning
#
# peek = learning()
# thresh_results, acc_results, num_results = peek.feature_selection_with_model(models, X_train, Y_train, X_test, Y_test)
#
# st.write(thresh_results)
# st.write(num_results)
# st.write(acc_results)

# dict_index = { 'XGBClassifier': 12, 'RandomForestClassifier': 4, 'GradientBoostingClassifier': 16}
# y_prob_dict, y_pred_matrix = peek.feature_selection_with_model(models, X_sub_train, Y1_sub_train, df_submit_x, dict_index = dict_index, thresh_results = thresh_results, num_classes = 2, verbose = 0)
# st.write(y_prob_dict['Ensamble'])
# y_prob_dict['Ensamble'][1].to_csv('sub1_feature_selection.csv', index = False)
# df = pd.read_csv('sub1_feature_selection.csv')
# st.write(df)

# dict_index = { 'XGBClassifier': 23, 'RandomForestClassifier': 7, 'GradientBoostingClassifier': 16}
# y_prob_dict, y_pred_matrix = peek.feature_selection_with_model(models, X_sub_train, Y2_sub_train, df_submit_x, dict_index = dict_index, thresh_results = thresh_results, num_classes = 2, verbose = 0)
# st.write(y_prob_dict['Ensamble'])
# y_prob_dict['Ensamble'][1].to_csv('sub2_feature_selection.csv', index = False)
# df = pd.read_csv('sub2_feature_selection.csv')
# st.write(df)

########################################################

# df1 = pd.read_csv('sub1_feature_selection.csv')
# st.write(df1)
# df2 = pd.read_csv('sub2_feature_selection.csv')
# st.write(df2)
#
# df_sub['seasonal_vaccine'] = df1
# df_sub['h1n1_vaccine'] = df2
#
# st.write(df_sub)
# df_sub.to_csv('sub_featur_select.csv', index = False)
# df = pd.read_csv('sub_featur_select.csv')
# st.write(df)


# ███████ ██    ██ ██████  ███    ███ ██ ███████ ███████ ██ ███    ██  ██████
# ██      ██    ██ ██   ██ ████  ████ ██ ██      ██      ██ ████   ██ ██    ██
# ███████ ██    ██ ██████  ██ ████ ██ ██ ███████ ███████ ██ ██ ██  ██ ██    ██
#      ██ ██    ██ ██   ██ ██  ██  ██ ██      ██      ██ ██ ██  ██ ██ ██    ██
# ███████  ██████  ██████  ██      ██ ██ ███████ ███████ ██ ██   ████  ██████



# learn1 = learning_class()
# models1 = learn1.get_models(['XGBClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier'])
# models1['deep learning normal binary'] = learn1.get_deep_learning_model(input_dl = X_sub_train.shape[1], net_type = 'normal', loss_type = 'binary')
#
# # X_train_samp, X_train_val, Y_train_samp, Y_train_val = train_test_split(X_sub_train, Y1_sub_train, test_size=0.1, random_state = 222)
# # models1 = learn1.train_models(models1, X_train_samp, Y_train_samp, epochs = 10, validation_data = [( X_train_samp, Y_train_samp),(X_train_val, Y_train_val)])[0]
# models1 = learn1.train_models(models1, X_sub_train, Y1_sub_train, epochs = 50)[0]
#
# learn2 = learning_class()
# models2 = learn2.get_models(['XGBClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier'])
# models2['deep learning normal binary'] = learn1.get_deep_learning_model(input_dl = X_sub_train.shape[1], net_type = 'normal', loss_type = 'binary')
#
# # X_train_samp, X_train_val, Y_train_samp, Y_train_val = train_test_split(X_sub_train, Y2_sub_train, test_size=0.1, random_state = 222)
# # models2 = learn1.train_models(models2, X_train_samp, Y_train_samp, epochs = 10, validation_data = [( X_train_samp, Y_train_samp),(X_train_val, Y_train_val)])[0]
# models2 = learn1.train_models(models2, X_sub_train, Y2_sub_train, epochs = 50)[0]
#
# #
# #
# Y1_sub, _ = learn1.prob_matrix_generator(models1, df_submit_x, 2)
# df_sub['seasonal_vaccine'] = Y1_sub['Ensamble'][1]
#
# Y2_sub, _ = learn2.prob_matrix_generator(models2, df_submit_x, 2)
# df_sub['h1n1_vaccine'] = Y2_sub['Ensamble'][1]
#
#
# st.write(df_sub)
# # df_sub.to_csv('sub_file_temp.csv', index = False)
# # df = pd.read_csv('sub_file_temp.csv')
# df_sub.to_csv('sub_dnn_and_other.csv', index = False)
# df = pd.read_csv('sub_dnn_and_other.csv')
# st.write(df)

# best so far hyperparameters tune (no early stop)
