from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import sys

from vrae.vrae import VRAE
from vrae.utils import *
import torch
import plotly
from torch.utils.data import DataLoader, TensorDataset

from model import reliable_predict


def main():
    source_file, output_path = sys.argv[1:]
    
    bins_path = "./nn_bins.pickle"
    model_path = "./nn_weights.ckpt"
    
    result = reliable_predict(source_file, bins_path, model_path)
    
    
    n_amt = 10  # сколько максимальных и минимальных сумм транзакций заменяем по каждому коду
    n_mcc = 10  # оставляем только коды с частотностью > n_mcc


    df_transactions = pd.read_csv(source_file)

    user_id = np.array(df_transactions['user_id'])
    mcc_code = np.array(df_transactions['mcc_code'])
    transaction_amt = np.array(df_transactions['transaction_amt'])


    for u_id in np.unique(user_id):

        amt_user = transaction_amt[user_id == u_id]
        mcc_user = mcc_code[user_id == u_id]

        mcc_unique = np.unique(mcc_user, return_counts = True)
        mcc_freq = mcc_unique[1]
        mcc_uniq = mcc_unique[0][mcc_unique[1] > n_mcc]
        mcc_unique = mcc_unique[0]
        mcc_freq_i, = np.where(mcc_uniq == mcc_unique[np.argmax(mcc_freq)])[0]  # индекс наиболее частотного кода в mcc_uniq   

        # чистим аномальные суммы транзакций для кодов, которые встречаются более n_amt раз

        amt_f_mcc = np.array([None] * len(mcc_uniq))
        p_perc = np.array([None] * len(mcc_uniq))

        for m, mcc in enumerate(mcc_uniq):     
          amt_f_mcc[m] = amt_user[mcc_user == mcc]  # все суммы транзакций по каждому коду
          p_perc[m] = 100 * n_amt / len(amt_f_mcc[m])  # перцентиль, за которым лежат аномальные значения
          if p_perc[m] > 100:
            p_perc[m] = 100
          elif p_perc[m] < 0:
            p_perc[m] = 0

        if n_amt > 0:
          for i in range(len(amt_user)):  # заменяем на медианы
            if mcc_user[i] in mcc_uniq:
              amt_t = amt_f_mcc[mcc_uniq == mcc_user[i]][0]
              p = p_perc[mcc_uniq == mcc_user[i]][0]

              if amt_user[i] > np.percentile(amt_t, 100 - p) or amt_user[i] < np.percentile(amt_t, p):
                amt_user[i] = np.median(amt_t)

        # чистим коды, которые встречаются менее n_mcc раз, заменяем на наиболее частотный

        if n_mcc > 0:
          amt_freq_med = np.median(amt_f_mcc[mcc_freq_i])
          for i in range(len(mcc_user)):
            if mcc_user[i] not in mcc_uniq:
                mcc_user[i] = mcc_uniq[mcc_freq_i]
                amt_user[i] = amt_freq_med
            
        mcc_code[user_id == u_id] = mcc_user
        transaction_amt[user_id == u_id] = amt_user

    df_transactions['mcc_code'] = mcc_code
    df_transactions['transaction_amt'] = transaction_amt
    
    source_file = './transactions_' + str(n_amt) + '_' + str(n_mcc) + '.csv'
    df_transactions.to_csv(source_file, index=False)
    
    
    #   --------------  VRAE
    
    transactions_vrae = pd.read_csv(source_file, parse_dates=['transaction_dttm']).assign(
            hour_of_day=lambda x: x.transaction_dttm.dt.hour,
            day_of_week=lambda x: x.transaction_dttm.dt.dayofweek,
            day_of_month=lambda x: x.transaction_dttm.dt.day,
            month=lambda x: x.transaction_dttm.dt.month)

    transactions_vrae['timediff'] = pd.to_timedelta(transactions_vrae['transaction_dttm'] - transactions_vrae['transaction_dttm'][0]).astype('timedelta64[h]')
    transactions_vrae['time_day'] = transactions_vrae.transaction_dttm.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second) # секунд с начала дня
    transactions_vrae['debit'] = transactions_vrae.loc[transactions_vrae['transaction_amt'] > 0, 'transaction_amt']
    transactions_vrae['credit'] = transactions_vrae.loc[transactions_vrae['transaction_amt'] < 0, 'transaction_amt']

    transactions_vrae.index = transactions_vrae['user_id']
    transactions_vrae = transactions_vrae.drop(columns=['user_id', 'transaction_dttm'])
    transactions_vrae = pd.get_dummies(transactions_vrae, columns=['mcc_code', 'currency_rk', 'hour_of_day', 'day_of_week', 'day_of_month', 'month'])

    cols = ['transaction_amt', 'timediff', 'time_day', 'debit', 'credit', 'mcc_code_-1', 'mcc_code_742', 'mcc_code_763', 'mcc_code_780', 'mcc_code_1520', 'mcc_code_1711', 'mcc_code_1731', 'mcc_code_1740', 'mcc_code_1750', 'mcc_code_1761', 'mcc_code_1799', 'mcc_code_2741', 'mcc_code_2791', 'mcc_code_2842', 'mcc_code_3005', 'mcc_code_3008', 'mcc_code_3009', 'mcc_code_3011', 'mcc_code_3015', 'mcc_code_3034', 'mcc_code_3035', 'mcc_code_3042', 'mcc_code_3047', 'mcc_code_3048', 'mcc_code_3068', 'mcc_code_3211', 'mcc_code_3217', 'mcc_code_3245', 'mcc_code_3256', 'mcc_code_3301', 'mcc_code_3357', 'mcc_code_3501', 'mcc_code_3503', 'mcc_code_3504', 'mcc_code_3509', 'mcc_code_3530', 'mcc_code_3533', 'mcc_code_3543', 'mcc_code_3553', 'mcc_code_3583', 'mcc_code_3586', 'mcc_code_3640', 'mcc_code_3641', 'mcc_code_3643', 'mcc_code_3649', 'mcc_code_3655', 'mcc_code_3665', 'mcc_code_3690', 'mcc_code_3692', 'mcc_code_3750', 'mcc_code_4011', 'mcc_code_4111', 'mcc_code_4112', 'mcc_code_4119', 'mcc_code_4121', 'mcc_code_4131', 'mcc_code_4214', 'mcc_code_4215', 'mcc_code_4225', 'mcc_code_4411', 'mcc_code_4457', 'mcc_code_4511', 'mcc_code_4582', 'mcc_code_4722', 'mcc_code_4784', 'mcc_code_4789', 'mcc_code_4812', 'mcc_code_4814', 'mcc_code_4816', 'mcc_code_4829', 'mcc_code_4899', 'mcc_code_4900', 'mcc_code_5013', 'mcc_code_5021', 'mcc_code_5039', 'mcc_code_5044', 'mcc_code_5045', 'mcc_code_5046', 'mcc_code_5047', 'mcc_code_5051', 'mcc_code_5065', 'mcc_code_5072', 'mcc_code_5074', 'mcc_code_5085', 'mcc_code_5094', 'mcc_code_5099', 'mcc_code_5111', 'mcc_code_5122', 'mcc_code_5131', 'mcc_code_5137', 'mcc_code_5139', 'mcc_code_5169', 'mcc_code_5172', 'mcc_code_5192', 'mcc_code_5193', 'mcc_code_5198', 'mcc_code_5199', 'mcc_code_5200', 'mcc_code_5211', 'mcc_code_5231', 'mcc_code_5251', 'mcc_code_5261', 'mcc_code_5300', 'mcc_code_5309', 'mcc_code_5310', 'mcc_code_5311', 'mcc_code_5331', 'mcc_code_5399', 'mcc_code_5411', 'mcc_code_5422', 'mcc_code_5441', 'mcc_code_5451', 'mcc_code_5462', 'mcc_code_5499', 'mcc_code_5511', 'mcc_code_5521', 'mcc_code_5531', 'mcc_code_5532', 'mcc_code_5533', 'mcc_code_5541', 'mcc_code_5542', 'mcc_code_5551', 'mcc_code_5561', 'mcc_code_5571', 'mcc_code_5599', 'mcc_code_5611', 'mcc_code_5621', 'mcc_code_5631', 'mcc_code_5641', 'mcc_code_5651', 'mcc_code_5655', 'mcc_code_5661', 'mcc_code_5681', 'mcc_code_5691', 'mcc_code_5697', 'mcc_code_5698', 'mcc_code_5699', 'mcc_code_5712', 'mcc_code_5713', 'mcc_code_5714', 'mcc_code_5718', 'mcc_code_5719', 'mcc_code_5722', 'mcc_code_5732', 'mcc_code_5733', 'mcc_code_5734', 'mcc_code_5735', 'mcc_code_5811', 'mcc_code_5812', 'mcc_code_5813', 'mcc_code_5814', 'mcc_code_5815', 'mcc_code_5816', 'mcc_code_5817', 'mcc_code_5818', 'mcc_code_5912', 'mcc_code_5921', 'mcc_code_5931', 'mcc_code_5932', 'mcc_code_5933', 'mcc_code_5937', 'mcc_code_5940', 'mcc_code_5941', 'mcc_code_5942', 'mcc_code_5943', 'mcc_code_5944', 'mcc_code_5945', 'mcc_code_5946', 'mcc_code_5947', 'mcc_code_5948', 'mcc_code_5949', 'mcc_code_5950', 'mcc_code_5960', 'mcc_code_5963', 'mcc_code_5964', 'mcc_code_5965', 'mcc_code_5967', 'mcc_code_5968', 'mcc_code_5969', 'mcc_code_5970', 'mcc_code_5971', 'mcc_code_5972', 'mcc_code_5973', 'mcc_code_5975', 'mcc_code_5976', 'mcc_code_5977', 'mcc_code_5978', 'mcc_code_5983', 'mcc_code_5992', 'mcc_code_5993', 'mcc_code_5994', 'mcc_code_5995', 'mcc_code_5996', 'mcc_code_5998', 'mcc_code_5999', 'mcc_code_6010', 'mcc_code_6011', 'mcc_code_6012', 'mcc_code_6050', 'mcc_code_6051', 'mcc_code_6211', 'mcc_code_6300', 'mcc_code_6399', 'mcc_code_6513', 'mcc_code_6532', 'mcc_code_6536', 'mcc_code_6537', 'mcc_code_6538', 'mcc_code_6540', 'mcc_code_6555', 'mcc_code_7011', 'mcc_code_7012', 'mcc_code_7032', 'mcc_code_7033', 'mcc_code_7210', 'mcc_code_7211', 'mcc_code_7216', 'mcc_code_7221', 'mcc_code_7230', 'mcc_code_7251', 'mcc_code_7261', 'mcc_code_7273', 'mcc_code_7276', 'mcc_code_7277', 'mcc_code_7278', 'mcc_code_7296', 'mcc_code_7297', 'mcc_code_7298', 'mcc_code_7299', 'mcc_code_7311', 'mcc_code_7321', 'mcc_code_7333', 'mcc_code_7338', 'mcc_code_7342', 'mcc_code_7349', 'mcc_code_7361', 'mcc_code_7372', 'mcc_code_7375', 'mcc_code_7379', 'mcc_code_7392', 'mcc_code_7393', 'mcc_code_7394', 'mcc_code_7395', 'mcc_code_7399', 'mcc_code_7512', 'mcc_code_7519', 'mcc_code_7523', 'mcc_code_7531', 'mcc_code_7534', 'mcc_code_7535', 'mcc_code_7538', 'mcc_code_7542', 'mcc_code_7549', 'mcc_code_7622', 'mcc_code_7623', 'mcc_code_7629', 'mcc_code_7631', 'mcc_code_7641', 'mcc_code_7699', 'mcc_code_7829', 'mcc_code_7832', 'mcc_code_7841', 'mcc_code_7911', 'mcc_code_7922', 'mcc_code_7929', 'mcc_code_7932', 'mcc_code_7933', 'mcc_code_7941', 'mcc_code_7991', 'mcc_code_7992', 'mcc_code_7993', 'mcc_code_7994', 'mcc_code_7995', 'mcc_code_7996', 'mcc_code_7997', 'mcc_code_7998', 'mcc_code_7999', 'mcc_code_8011', 'mcc_code_8021', 'mcc_code_8041', 'mcc_code_8042', 'mcc_code_8043', 'mcc_code_8049', 'mcc_code_8050', 'mcc_code_8062', 'mcc_code_8071', 'mcc_code_8099', 'mcc_code_8111', 'mcc_code_8211', 'mcc_code_8220', 'mcc_code_8241', 'mcc_code_8244', 'mcc_code_8249', 'mcc_code_8299', 'mcc_code_8351', 'mcc_code_8398', 'mcc_code_8641', 'mcc_code_8661', 'mcc_code_8675', 'mcc_code_8699', 'mcc_code_8734', 'mcc_code_8911', 'mcc_code_8931', 'mcc_code_8999', 'mcc_code_9211', 'mcc_code_9222', 'mcc_code_9311', 'mcc_code_9399', 'mcc_code_9402', 'currency_rk_48', 'currency_rk_50', 'currency_rk_60', 'hour_of_day_0', 'hour_of_day_1', 'hour_of_day_2', 'hour_of_day_3', 'hour_of_day_4', 'hour_of_day_5', 'hour_of_day_6', 'hour_of_day_7', 'hour_of_day_8', 'hour_of_day_9', 'hour_of_day_10', 'hour_of_day_11', 'hour_of_day_12', 'hour_of_day_13', 'hour_of_day_14', 'hour_of_day_15', 'hour_of_day_16', 'hour_of_day_17', 'hour_of_day_18', 'hour_of_day_19', 'hour_of_day_20', 'hour_of_day_21', 'hour_of_day_22', 'hour_of_day_23', 'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6', 'day_of_month_1', 'day_of_month_2', 'day_of_month_3', 'day_of_month_4', 'day_of_month_5', 'day_of_month_6', 'day_of_month_7', 'day_of_month_8', 'day_of_month_9', 'day_of_month_10', 'day_of_month_11', 'day_of_month_12', 'day_of_month_13', 'day_of_month_14', 'day_of_month_15', 'day_of_month_16', 'day_of_month_17', 'day_of_month_18', 'day_of_month_19', 'day_of_month_20', 'day_of_month_21', 'day_of_month_22', 'day_of_month_23', 'day_of_month_24', 'day_of_month_25', 'day_of_month_26', 'day_of_month_27', 'day_of_month_28', 'day_of_month_29', 'day_of_month_30', 'day_of_month_31', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']
    transactions_vrae = transactions_vrae.reindex(columns=cols)
    transactions_vrae = transactions_vrae[cols]
    transactions_vrae = transactions_vrae.fillna(0)

    X_train = []
    u_index = np.unique(transactions_vrae.index)
    for u in u_index:
        df_temp = transactions_vrae[transactions_vrae.index == u].copy()
        scaler = StandardScaler().fit(df_temp.loc[:, ['transaction_amt', 'debit', 'credit']])
        df_temp.loc[:, ['transaction_amt', 'debit', 'credit']] = scaler.transform(df_temp.loc[:, ['transaction_amt', 'debit', 'credit']])
        df_temp.loc[:, 'transaction_cum_sum'] = df_temp['transaction_amt'].cumsum()
        df_temp.loc[:, 'debit_cum_sum'] = df_temp['debit'].cumsum()
        df_temp.loc[:, 'credit_cum_sum'] = df_temp['credit'].cumsum()
        df_temp.index = list(range(len(df_temp)))
        df_temp.loc[:, 'timediff'] = df_temp['timediff'].subtract(df_temp.loc[:, 'timediff'][0])
        X_train.append(np.array(df_temp))
        gar_lst = [df_temp]
        del df_temp, gar_lst
    gar_lst = [transactions_vrae]
    del transactions_vrae, gar_lst
    X_train = np.array(X_train)

    orig_len = len(X_train)
    pad_len = (32 - len(X_train) % 32)
    if pad_len < 32:
      random_indices = np.random.randint(0, len(X_train), pad_len)
      random_values = X_train[random_indices]
      X_train = np.concatenate((X_train, random_values))

    train_dataset = TensorDataset(torch.from_numpy(X_train))
    sequence_length = X_train.shape[1]
    number_of_features = X_train.shape[2]
    #del X_train
    
    
    hidden_size = 120
    hidden_layer_depth = 1
    latent_length = 40
    batch_size = 32
    learning_rate = 0.0005
    n_epochs = 40
    dropout_rate = 0.1
    optimizer = 'Adam' # options: ADAM, SGD
    cuda = True # options: True, False
    print_every=30
    clip = True # options: True, False
    max_grad_norm=5
    loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
    block = 'LSTM' # options: LSTM, GRU
    
    if len(X_train) >= batch_size:     
        
        vrae = VRAE(sequence_length=sequence_length,
                    number_of_features = number_of_features,
                    hidden_size = hidden_size, 
                    hidden_layer_depth = hidden_layer_depth,
                    latent_length = latent_length,
                    batch_size = batch_size,
                    learning_rate = learning_rate,
                    n_epochs = n_epochs,
                    dropout_rate = dropout_rate,
                    optimizer = optimizer, 
                    cuda = cuda,
                    print_every=print_every, 
                    clip=clip, 
                    max_grad_norm=max_grad_norm,
                    loss = loss,
                    block = block,
                    dload = './model_dir')

        vrae.load('./model_dir/vrae_5896.pth')

        z_run = vrae.transform(train_dataset)
        df_z = pd.DataFrame(z_run[:orig_len])
        df_z.index = u_index[:len(df_z)]
    
    #  ----------------
    
    
    transactions = pd.read_csv(source_file, parse_dates=['transaction_dttm']).assign(
            hour_of_day=lambda x: x.transaction_dttm.dt.hour,
            day_of_week=lambda x: x.transaction_dttm.dt.dayofweek,
            day_of_month=lambda x: x.transaction_dttm.dt.day,
            month=lambda x: x.transaction_dttm.dt.month)

    transactions['timediff'] = pd.to_timedelta(transactions['transaction_dttm'] - transactions['transaction_dttm'][0]).astype('timedelta64[h]')
    transactions['time_day'] = transactions.transaction_dttm.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second) # секунд с начала дня
    transactions['debit'] = transactions.loc[transactions['transaction_amt'] > 0, 'transaction_amt']
    transactions['credit'] = transactions.loc[transactions['transaction_amt'] < 0, 'transaction_amt']

    sli = 20
    transactions = transactions.groupby('user_id').apply(lambda x: x.iloc[sli:-sli])
    transactions.index = list(range(transactions.shape[0]))

    u_ids = np.unique(transactions['user_id'])
    transactions = transactions.drop(transactions[transactions['mcc_code'] == 6012].index)
    u_drop = np.setdiff1d(u_ids, np.unique(transactions['user_id']))
    
    
    # извлечение признаков для CatBoost
    
    def get_stats(df, str_name):

        df[str_name + '_mean'] = mcc_freq[mcc_freq > 0].mean(axis=1)
        df[str_name + '_median'] = mcc_freq[mcc_freq > 0].median(axis=1)

        df[str_name + '_max'] = mcc_freq[mcc_freq > 0].max(axis=1)
        df[str_name + '_min'] = mcc_freq[mcc_freq > 0].min(axis=1)

        df[str_name + '_std'] = mcc_freq[mcc_freq > 0].std(axis=1)
        df[str_name + '_sem'] = mcc_freq[mcc_freq > 0].sem(axis=1)
        df[str_name + '_skew'] = mcc_freq[mcc_freq > 0].skew(axis=1)
        df[str_name + '_var'] = mcc_freq[mcc_freq > 0].var(axis=1)

        df[str_name + '_amplitude1'] = df[str_name + '_max'] - df[str_name + '_min']
        df[str_name + '_amplitude2'] = df[str_name + '_max'] - df[str_name + '_median']
        df[str_name + '_amplitude3'] = df[str_name + '_max'] - df[str_name + '_mean']
        df[str_name + '_amplitude4'] = df[str_name + '_median'] - df[str_name + '_min']
        df[str_name + '_amplitude5'] = df[str_name + '_mean'] - df[str_name + '_min']

        df[str_name + '_divmm1'] = df[str_name + '_max'] / df[str_name + '_min']
        df[str_name + '_divmm2'] = df[str_name + '_max'] / df[str_name + '_mean']
        df[str_name + '_divmm3'] = df[str_name + '_max'] / df[str_name + '_median']
        df[str_name + '_divmm4'] = df[str_name + '_mean'] / df[str_name + '_min']
        df[str_name + '_divmm5'] = df[str_name + '_median'] / df[str_name + '_min']

        df = df.fillna(0)

        return df

    # ------------

    mcc_freq = transactions.pivot_table(
        index='user_id', columns=['mcc_code'], values=['transaction_amt'], 
        aggfunc=['count'], fill_value=0)
    mcc_freq.columns = [f'mcc_freq-{i[2]}' for i in mcc_freq.columns]

    cols = ['mcc_freq--1', 'mcc_freq-742', 'mcc_freq-780', 'mcc_freq-2741', 'mcc_freq-3011', 'mcc_freq-4111', 'mcc_freq-4112', 'mcc_freq-4121', 'mcc_freq-4131', 'mcc_freq-4214', 'mcc_freq-4215', 'mcc_freq-4511', 'mcc_freq-4722', 'mcc_freq-4784', 'mcc_freq-4789', 'mcc_freq-4812', 'mcc_freq-4814', 'mcc_freq-4816', 'mcc_freq-4829', 'mcc_freq-4899', 'mcc_freq-4900', 'mcc_freq-5013', 'mcc_freq-5039', 'mcc_freq-5045', 'mcc_freq-5047', 'mcc_freq-5072', 'mcc_freq-5111', 'mcc_freq-5122', 'mcc_freq-5192', 'mcc_freq-5200', 'mcc_freq-5211', 'mcc_freq-5231', 'mcc_freq-5251', 'mcc_freq-5261', 'mcc_freq-5309', 'mcc_freq-5310', 'mcc_freq-5311', 'mcc_freq-5331', 'mcc_freq-5399', 'mcc_freq-5411', 'mcc_freq-5422', 'mcc_freq-5441', 'mcc_freq-5451', 'mcc_freq-5462', 'mcc_freq-5499', 'mcc_freq-5511', 'mcc_freq-5533', 'mcc_freq-5541', 'mcc_freq-5542', 'mcc_freq-5611', 'mcc_freq-5621', 'mcc_freq-5631', 'mcc_freq-5641', 'mcc_freq-5651', 'mcc_freq-5655', 'mcc_freq-5661', 'mcc_freq-5691', 'mcc_freq-5699', 'mcc_freq-5712', 'mcc_freq-5714', 'mcc_freq-5719', 'mcc_freq-5722', 'mcc_freq-5732', 'mcc_freq-5734', 'mcc_freq-5735', 'mcc_freq-5811', 'mcc_freq-5812', 'mcc_freq-5813', 'mcc_freq-5814', 'mcc_freq-5815', 'mcc_freq-5816', 'mcc_freq-5817', 'mcc_freq-5818', 'mcc_freq-5912', 'mcc_freq-5921', 'mcc_freq-5931', 'mcc_freq-5941', 'mcc_freq-5942', 'mcc_freq-5943', 'mcc_freq-5944', 'mcc_freq-5945', 'mcc_freq-5947', 'mcc_freq-5948', 'mcc_freq-5949', 'mcc_freq-5964', 'mcc_freq-5965', 'mcc_freq-5967', 'mcc_freq-5968', 'mcc_freq-5969', 'mcc_freq-5970', 'mcc_freq-5973', 'mcc_freq-5977', 'mcc_freq-5983', 'mcc_freq-5992', 'mcc_freq-5993', 'mcc_freq-5994', 'mcc_freq-5995', 'mcc_freq-5999', 'mcc_freq-6010', 'mcc_freq-6011', 'mcc_freq-6051', 'mcc_freq-6211', 'mcc_freq-6300', 'mcc_freq-6399', 'mcc_freq-6513', 'mcc_freq-6536', 'mcc_freq-6538', 'mcc_freq-6540', 'mcc_freq-7011', 'mcc_freq-7216', 'mcc_freq-7230', 'mcc_freq-7273', 'mcc_freq-7278', 'mcc_freq-7298', 'mcc_freq-7299', 'mcc_freq-7311', 'mcc_freq-7333', 'mcc_freq-7338', 'mcc_freq-7372', 'mcc_freq-7392', 'mcc_freq-7399', 'mcc_freq-7512', 'mcc_freq-7519', 'mcc_freq-7523', 'mcc_freq-7538', 'mcc_freq-7542', 'mcc_freq-7622', 'mcc_freq-7832', 'mcc_freq-7841', 'mcc_freq-7922', 'mcc_freq-7941', 'mcc_freq-7994', 'mcc_freq-7995', 'mcc_freq-7996', 'mcc_freq-7997', 'mcc_freq-7999', 'mcc_freq-8011', 'mcc_freq-8021', 'mcc_freq-8050', 'mcc_freq-8062', 'mcc_freq-8071', 'mcc_freq-8099', 'mcc_freq-8220', 'mcc_freq-8299', 'mcc_freq-8351', 'mcc_freq-8398', 'mcc_freq-8641', 'mcc_freq-8661', 'mcc_freq-8999', 'mcc_freq-9222', 'mcc_freq-9311', 'mcc_freq-9399', 'mcc_freq-9402']
    mcc_freq = mcc_freq.reindex(columns=cols)
    mcc_freq = mcc_freq[cols]
    mcc_freq = mcc_freq.fillna(0)

    for col in mcc_freq.columns:
        mcc_freq[col] //= 20

    mcc_freq_stats = pd.DataFrame()
    mcc_freq_stats.index = mcc_freq.index
    mcc_freq_stats['mcc_freq_stats_sum'] = mcc_freq.sum(axis=1)
    mcc_freq_stats = get_stats(mcc_freq_stats, 'mcc_freq_stats')

    # ------------

    mcc_freq_proc = mcc_freq.div(mcc_freq.sum(axis=1), axis=0).fillna(0) # процентный вид, сумма строки = 1
    mcc_freq_proc.columns = [f'mcc_freq_proc-{i}' for i in mcc_freq_proc.columns]

    cols = ['mcc_freq_proc-mcc_freq--1', 'mcc_freq_proc-mcc_freq-742', 'mcc_freq_proc-mcc_freq-780', 'mcc_freq_proc-mcc_freq-2741', 'mcc_freq_proc-mcc_freq-3011', 'mcc_freq_proc-mcc_freq-4111', 'mcc_freq_proc-mcc_freq-4112', 'mcc_freq_proc-mcc_freq-4121', 'mcc_freq_proc-mcc_freq-4131', 'mcc_freq_proc-mcc_freq-4214', 'mcc_freq_proc-mcc_freq-4215', 'mcc_freq_proc-mcc_freq-4511', 'mcc_freq_proc-mcc_freq-4722', 'mcc_freq_proc-mcc_freq-4784', 'mcc_freq_proc-mcc_freq-4789', 'mcc_freq_proc-mcc_freq-4812', 'mcc_freq_proc-mcc_freq-4814', 'mcc_freq_proc-mcc_freq-4816', 'mcc_freq_proc-mcc_freq-4829', 'mcc_freq_proc-mcc_freq-4899', 'mcc_freq_proc-mcc_freq-4900', 'mcc_freq_proc-mcc_freq-5013', 'mcc_freq_proc-mcc_freq-5039', 'mcc_freq_proc-mcc_freq-5045', 'mcc_freq_proc-mcc_freq-5047', 'mcc_freq_proc-mcc_freq-5072', 'mcc_freq_proc-mcc_freq-5111', 'mcc_freq_proc-mcc_freq-5122', 'mcc_freq_proc-mcc_freq-5192', 'mcc_freq_proc-mcc_freq-5200', 'mcc_freq_proc-mcc_freq-5211', 'mcc_freq_proc-mcc_freq-5231', 'mcc_freq_proc-mcc_freq-5251', 'mcc_freq_proc-mcc_freq-5261', 'mcc_freq_proc-mcc_freq-5309', 'mcc_freq_proc-mcc_freq-5310', 'mcc_freq_proc-mcc_freq-5311', 'mcc_freq_proc-mcc_freq-5331', 'mcc_freq_proc-mcc_freq-5399', 'mcc_freq_proc-mcc_freq-5411', 'mcc_freq_proc-mcc_freq-5422', 'mcc_freq_proc-mcc_freq-5441', 'mcc_freq_proc-mcc_freq-5451', 'mcc_freq_proc-mcc_freq-5462', 'mcc_freq_proc-mcc_freq-5499', 'mcc_freq_proc-mcc_freq-5511', 'mcc_freq_proc-mcc_freq-5533', 'mcc_freq_proc-mcc_freq-5541', 'mcc_freq_proc-mcc_freq-5542', 'mcc_freq_proc-mcc_freq-5611', 'mcc_freq_proc-mcc_freq-5621', 'mcc_freq_proc-mcc_freq-5631', 'mcc_freq_proc-mcc_freq-5641', 'mcc_freq_proc-mcc_freq-5651', 'mcc_freq_proc-mcc_freq-5655', 'mcc_freq_proc-mcc_freq-5661', 'mcc_freq_proc-mcc_freq-5691', 'mcc_freq_proc-mcc_freq-5699', 'mcc_freq_proc-mcc_freq-5712', 'mcc_freq_proc-mcc_freq-5714', 'mcc_freq_proc-mcc_freq-5719', 'mcc_freq_proc-mcc_freq-5722', 'mcc_freq_proc-mcc_freq-5732', 'mcc_freq_proc-mcc_freq-5734', 'mcc_freq_proc-mcc_freq-5735', 'mcc_freq_proc-mcc_freq-5811', 'mcc_freq_proc-mcc_freq-5812', 'mcc_freq_proc-mcc_freq-5813', 'mcc_freq_proc-mcc_freq-5814', 'mcc_freq_proc-mcc_freq-5815', 'mcc_freq_proc-mcc_freq-5816', 'mcc_freq_proc-mcc_freq-5817', 'mcc_freq_proc-mcc_freq-5818', 'mcc_freq_proc-mcc_freq-5912', 'mcc_freq_proc-mcc_freq-5921', 'mcc_freq_proc-mcc_freq-5931', 'mcc_freq_proc-mcc_freq-5941', 'mcc_freq_proc-mcc_freq-5942', 'mcc_freq_proc-mcc_freq-5943', 'mcc_freq_proc-mcc_freq-5944', 'mcc_freq_proc-mcc_freq-5945', 'mcc_freq_proc-mcc_freq-5947', 'mcc_freq_proc-mcc_freq-5948', 'mcc_freq_proc-mcc_freq-5949', 'mcc_freq_proc-mcc_freq-5964', 'mcc_freq_proc-mcc_freq-5965', 'mcc_freq_proc-mcc_freq-5967', 'mcc_freq_proc-mcc_freq-5968', 'mcc_freq_proc-mcc_freq-5969', 'mcc_freq_proc-mcc_freq-5970', 'mcc_freq_proc-mcc_freq-5973', 'mcc_freq_proc-mcc_freq-5977', 'mcc_freq_proc-mcc_freq-5983', 'mcc_freq_proc-mcc_freq-5992', 'mcc_freq_proc-mcc_freq-5993', 'mcc_freq_proc-mcc_freq-5994', 'mcc_freq_proc-mcc_freq-5995', 'mcc_freq_proc-mcc_freq-5999', 'mcc_freq_proc-mcc_freq-6010', 'mcc_freq_proc-mcc_freq-6011', 'mcc_freq_proc-mcc_freq-6051', 'mcc_freq_proc-mcc_freq-6211', 'mcc_freq_proc-mcc_freq-6300', 'mcc_freq_proc-mcc_freq-6399', 'mcc_freq_proc-mcc_freq-6513', 'mcc_freq_proc-mcc_freq-6536', 'mcc_freq_proc-mcc_freq-6538', 'mcc_freq_proc-mcc_freq-6540', 'mcc_freq_proc-mcc_freq-7011', 'mcc_freq_proc-mcc_freq-7216', 'mcc_freq_proc-mcc_freq-7230', 'mcc_freq_proc-mcc_freq-7273', 'mcc_freq_proc-mcc_freq-7278', 'mcc_freq_proc-mcc_freq-7298', 'mcc_freq_proc-mcc_freq-7299', 'mcc_freq_proc-mcc_freq-7311', 'mcc_freq_proc-mcc_freq-7333', 'mcc_freq_proc-mcc_freq-7338', 'mcc_freq_proc-mcc_freq-7372', 'mcc_freq_proc-mcc_freq-7392', 'mcc_freq_proc-mcc_freq-7399', 'mcc_freq_proc-mcc_freq-7512', 'mcc_freq_proc-mcc_freq-7519', 'mcc_freq_proc-mcc_freq-7523', 'mcc_freq_proc-mcc_freq-7538', 'mcc_freq_proc-mcc_freq-7542', 'mcc_freq_proc-mcc_freq-7622', 'mcc_freq_proc-mcc_freq-7832', 'mcc_freq_proc-mcc_freq-7841', 'mcc_freq_proc-mcc_freq-7922', 'mcc_freq_proc-mcc_freq-7941', 'mcc_freq_proc-mcc_freq-7994', 'mcc_freq_proc-mcc_freq-7995', 'mcc_freq_proc-mcc_freq-7996', 'mcc_freq_proc-mcc_freq-7997', 'mcc_freq_proc-mcc_freq-7999', 'mcc_freq_proc-mcc_freq-8011', 'mcc_freq_proc-mcc_freq-8021', 'mcc_freq_proc-mcc_freq-8050', 'mcc_freq_proc-mcc_freq-8062', 'mcc_freq_proc-mcc_freq-8071', 'mcc_freq_proc-mcc_freq-8099', 'mcc_freq_proc-mcc_freq-8220', 'mcc_freq_proc-mcc_freq-8299', 'mcc_freq_proc-mcc_freq-8351', 'mcc_freq_proc-mcc_freq-8398', 'mcc_freq_proc-mcc_freq-8641', 'mcc_freq_proc-mcc_freq-8661', 'mcc_freq_proc-mcc_freq-8999', 'mcc_freq_proc-mcc_freq-9222', 'mcc_freq_proc-mcc_freq-9311', 'mcc_freq_proc-mcc_freq-9399', 'mcc_freq_proc-mcc_freq-9402']
    mcc_freq_proc = mcc_freq_proc.reindex(columns=cols)
    mcc_freq_proc = mcc_freq_proc[cols]
    mcc_freq_proc = mcc_freq_proc.fillna(0)

    mcc_freq_proc_stats = pd.DataFrame()
    mcc_freq_proc_stats.index = mcc_freq.index
    mcc_freq_proc_stats = get_stats(mcc_freq_proc_stats, 'mcc_freq_proc_stats')

    # ------------

    mcc_proc = transactions.pivot_table(
        index='user_id', columns=['mcc_code'], values=['transaction_amt'], 
        aggfunc=[np.sum], fill_value=0)
    mcc_proc.columns = [f'mcc_proc-{i[2]}' for i in mcc_proc.columns]

    mcc_proc = mcc_proc.div(mcc_proc.sum(axis=1), axis=0).fillna(0) # процентный вид, сумма строки = 1

    cols = ['mcc_proc--1', 'mcc_proc-742', 'mcc_proc-780', 'mcc_proc-2741', 'mcc_proc-3011', 'mcc_proc-4111', 'mcc_proc-4112', 'mcc_proc-4121', 'mcc_proc-4131', 'mcc_proc-4214', 'mcc_proc-4215', 'mcc_proc-4511', 'mcc_proc-4722', 'mcc_proc-4784', 'mcc_proc-4789', 'mcc_proc-4812', 'mcc_proc-4814', 'mcc_proc-4816', 'mcc_proc-4829', 'mcc_proc-4899', 'mcc_proc-4900', 'mcc_proc-5013', 'mcc_proc-5039', 'mcc_proc-5045', 'mcc_proc-5047', 'mcc_proc-5072', 'mcc_proc-5111', 'mcc_proc-5122', 'mcc_proc-5192', 'mcc_proc-5200', 'mcc_proc-5211', 'mcc_proc-5231', 'mcc_proc-5251', 'mcc_proc-5261', 'mcc_proc-5309', 'mcc_proc-5310', 'mcc_proc-5311', 'mcc_proc-5331', 'mcc_proc-5399', 'mcc_proc-5411', 'mcc_proc-5422', 'mcc_proc-5441', 'mcc_proc-5451', 'mcc_proc-5462', 'mcc_proc-5499', 'mcc_proc-5511', 'mcc_proc-5533', 'mcc_proc-5541', 'mcc_proc-5542', 'mcc_proc-5611', 'mcc_proc-5621', 'mcc_proc-5631', 'mcc_proc-5641', 'mcc_proc-5651', 'mcc_proc-5655', 'mcc_proc-5661', 'mcc_proc-5691', 'mcc_proc-5699', 'mcc_proc-5712', 'mcc_proc-5714', 'mcc_proc-5719', 'mcc_proc-5722', 'mcc_proc-5732', 'mcc_proc-5734', 'mcc_proc-5735', 'mcc_proc-5811', 'mcc_proc-5812', 'mcc_proc-5813', 'mcc_proc-5814', 'mcc_proc-5815', 'mcc_proc-5816', 'mcc_proc-5817', 'mcc_proc-5818', 'mcc_proc-5912', 'mcc_proc-5921', 'mcc_proc-5931', 'mcc_proc-5941', 'mcc_proc-5942', 'mcc_proc-5943', 'mcc_proc-5944', 'mcc_proc-5945', 'mcc_proc-5947', 'mcc_proc-5948', 'mcc_proc-5949', 'mcc_proc-5964', 'mcc_proc-5965', 'mcc_proc-5967', 'mcc_proc-5968', 'mcc_proc-5969', 'mcc_proc-5970', 'mcc_proc-5973', 'mcc_proc-5977', 'mcc_proc-5983', 'mcc_proc-5992', 'mcc_proc-5993', 'mcc_proc-5994', 'mcc_proc-5995', 'mcc_proc-5999', 'mcc_proc-6010', 'mcc_proc-6011', 'mcc_proc-6051', 'mcc_proc-6211', 'mcc_proc-6300', 'mcc_proc-6399', 'mcc_proc-6513', 'mcc_proc-6536', 'mcc_proc-6538', 'mcc_proc-6540', 'mcc_proc-7011', 'mcc_proc-7216', 'mcc_proc-7230', 'mcc_proc-7273', 'mcc_proc-7278', 'mcc_proc-7298', 'mcc_proc-7299', 'mcc_proc-7311', 'mcc_proc-7333', 'mcc_proc-7338', 'mcc_proc-7372', 'mcc_proc-7392', 'mcc_proc-7399', 'mcc_proc-7512', 'mcc_proc-7519', 'mcc_proc-7523', 'mcc_proc-7538', 'mcc_proc-7542', 'mcc_proc-7622', 'mcc_proc-7832', 'mcc_proc-7841', 'mcc_proc-7922', 'mcc_proc-7941', 'mcc_proc-7994', 'mcc_proc-7995', 'mcc_proc-7996', 'mcc_proc-7997', 'mcc_proc-7999', 'mcc_proc-8011', 'mcc_proc-8021', 'mcc_proc-8050', 'mcc_proc-8062', 'mcc_proc-8071', 'mcc_proc-8099', 'mcc_proc-8220', 'mcc_proc-8299', 'mcc_proc-8351', 'mcc_proc-8398', 'mcc_proc-8641', 'mcc_proc-8661', 'mcc_proc-8999', 'mcc_proc-9222', 'mcc_proc-9311', 'mcc_proc-9399', 'mcc_proc-9402']
    mcc_proc = mcc_proc.reindex(columns=cols)
    mcc_proc = mcc_proc[cols]
    mcc_proc = mcc_proc.fillna(0)

    mcc_proc_stats = pd.DataFrame()
    mcc_proc_stats.index = mcc_proc.index
    mcc_proc_stats = get_stats(mcc_proc_stats, 'mcc_proc_stats')

    # ------------

    mcc_proc_deb = transactions.pivot_table(
        index='user_id', columns=['mcc_code'], values=['debit'], 
        aggfunc=[np.sum], fill_value=0)
    mcc_proc_deb.columns = [f'mcc_proc_deb-{i[2]}' for i in mcc_proc_deb.columns]

    debit_sum = mcc_proc_deb.sum(axis=1) # сумма доходов людей

    mcc_proc_deb = mcc_proc_deb.div(mcc_proc_deb.sum(axis=1), axis=0).fillna(0) # процентный вид, сумма строки = 1

    cols = ['mcc_proc_deb--1', 'mcc_proc_deb-742', 'mcc_proc_deb-780', 'mcc_proc_deb-2741', 'mcc_proc_deb-3011', 'mcc_proc_deb-4111', 'mcc_proc_deb-4112', 'mcc_proc_deb-4121', 'mcc_proc_deb-4131', 'mcc_proc_deb-4214', 'mcc_proc_deb-4215', 'mcc_proc_deb-4511', 'mcc_proc_deb-4722', 'mcc_proc_deb-4784', 'mcc_proc_deb-4789', 'mcc_proc_deb-4812', 'mcc_proc_deb-4814', 'mcc_proc_deb-4816', 'mcc_proc_deb-4829', 'mcc_proc_deb-4899', 'mcc_proc_deb-4900', 'mcc_proc_deb-5013', 'mcc_proc_deb-5039', 'mcc_proc_deb-5045', 'mcc_proc_deb-5047', 'mcc_proc_deb-5072', 'mcc_proc_deb-5111', 'mcc_proc_deb-5122', 'mcc_proc_deb-5192', 'mcc_proc_deb-5200', 'mcc_proc_deb-5211', 'mcc_proc_deb-5231', 'mcc_proc_deb-5251', 'mcc_proc_deb-5261', 'mcc_proc_deb-5309', 'mcc_proc_deb-5310', 'mcc_proc_deb-5311', 'mcc_proc_deb-5331', 'mcc_proc_deb-5399', 'mcc_proc_deb-5411', 'mcc_proc_deb-5422', 'mcc_proc_deb-5441', 'mcc_proc_deb-5451', 'mcc_proc_deb-5462', 'mcc_proc_deb-5499', 'mcc_proc_deb-5511', 'mcc_proc_deb-5533', 'mcc_proc_deb-5541', 'mcc_proc_deb-5542', 'mcc_proc_deb-5611', 'mcc_proc_deb-5621', 'mcc_proc_deb-5631', 'mcc_proc_deb-5641', 'mcc_proc_deb-5651', 'mcc_proc_deb-5655', 'mcc_proc_deb-5661', 'mcc_proc_deb-5691', 'mcc_proc_deb-5699', 'mcc_proc_deb-5712', 'mcc_proc_deb-5714', 'mcc_proc_deb-5719', 'mcc_proc_deb-5722', 'mcc_proc_deb-5732', 'mcc_proc_deb-5734', 'mcc_proc_deb-5735', 'mcc_proc_deb-5811', 'mcc_proc_deb-5812', 'mcc_proc_deb-5813', 'mcc_proc_deb-5814', 'mcc_proc_deb-5815', 'mcc_proc_deb-5816', 'mcc_proc_deb-5817', 'mcc_proc_deb-5818', 'mcc_proc_deb-5912', 'mcc_proc_deb-5921', 'mcc_proc_deb-5931', 'mcc_proc_deb-5941', 'mcc_proc_deb-5942', 'mcc_proc_deb-5943', 'mcc_proc_deb-5944', 'mcc_proc_deb-5945', 'mcc_proc_deb-5947', 'mcc_proc_deb-5948', 'mcc_proc_deb-5949', 'mcc_proc_deb-5964', 'mcc_proc_deb-5965', 'mcc_proc_deb-5967', 'mcc_proc_deb-5968', 'mcc_proc_deb-5969', 'mcc_proc_deb-5970', 'mcc_proc_deb-5973', 'mcc_proc_deb-5977', 'mcc_proc_deb-5983', 'mcc_proc_deb-5992', 'mcc_proc_deb-5993', 'mcc_proc_deb-5994', 'mcc_proc_deb-5995', 'mcc_proc_deb-5999', 'mcc_proc_deb-6010', 'mcc_proc_deb-6011', 'mcc_proc_deb-6051', 'mcc_proc_deb-6211', 'mcc_proc_deb-6300', 'mcc_proc_deb-6399', 'mcc_proc_deb-6513', 'mcc_proc_deb-6536', 'mcc_proc_deb-6538', 'mcc_proc_deb-6540', 'mcc_proc_deb-7011', 'mcc_proc_deb-7216', 'mcc_proc_deb-7230', 'mcc_proc_deb-7273', 'mcc_proc_deb-7278', 'mcc_proc_deb-7298', 'mcc_proc_deb-7299', 'mcc_proc_deb-7311', 'mcc_proc_deb-7333', 'mcc_proc_deb-7338', 'mcc_proc_deb-7372', 'mcc_proc_deb-7392', 'mcc_proc_deb-7399', 'mcc_proc_deb-7512', 'mcc_proc_deb-7519', 'mcc_proc_deb-7523', 'mcc_proc_deb-7538', 'mcc_proc_deb-7542', 'mcc_proc_deb-7622', 'mcc_proc_deb-7832', 'mcc_proc_deb-7841', 'mcc_proc_deb-7922', 'mcc_proc_deb-7941', 'mcc_proc_deb-7994', 'mcc_proc_deb-7995', 'mcc_proc_deb-7996', 'mcc_proc_deb-7997', 'mcc_proc_deb-7999', 'mcc_proc_deb-8011', 'mcc_proc_deb-8021', 'mcc_proc_deb-8050', 'mcc_proc_deb-8062', 'mcc_proc_deb-8071', 'mcc_proc_deb-8099', 'mcc_proc_deb-8220', 'mcc_proc_deb-8299', 'mcc_proc_deb-8351', 'mcc_proc_deb-8398', 'mcc_proc_deb-8641', 'mcc_proc_deb-8661', 'mcc_proc_deb-8999', 'mcc_proc_deb-9222', 'mcc_proc_deb-9311', 'mcc_proc_deb-9399', 'mcc_proc_deb-9402']
    mcc_proc_deb = mcc_proc_deb.reindex(columns=cols)
    mcc_proc_deb = mcc_proc_deb[cols]
    mcc_proc_deb = mcc_proc_deb.fillna(0)

    mcc_proc_deb_stats = pd.DataFrame()
    mcc_proc_deb_stats.index = mcc_proc_deb.index
    mcc_proc_deb_stats = get_stats(mcc_proc_deb_stats, 'mcc_proc_deb_stats')

    # ------------

    mcc_proc_cred = transactions.pivot_table(
        index='user_id', columns=['mcc_code'], values=['credit'], 
        aggfunc=[np.sum], fill_value=0)
    mcc_proc_cred.columns = [f'mcc_proc_cred-{i[2]}' for i in mcc_proc_cred.columns]

    credit_sum = mcc_proc_cred.sum(axis=1) # сумма доходов людей

    mcc_proc_cred = mcc_proc_cred.div(mcc_proc_cred.sum(axis=1), axis=0).fillna(0) # процентный вид, сумма строки = 1

    cols = ['mcc_proc_cred--1', 'mcc_proc_cred-742', 'mcc_proc_cred-780', 'mcc_proc_cred-2741', 'mcc_proc_cred-3011', 'mcc_proc_cred-4111', 'mcc_proc_cred-4112', 'mcc_proc_cred-4121', 'mcc_proc_cred-4131', 'mcc_proc_cred-4214', 'mcc_proc_cred-4215', 'mcc_proc_cred-4511', 'mcc_proc_cred-4722', 'mcc_proc_cred-4784', 'mcc_proc_cred-4789', 'mcc_proc_cred-4812', 'mcc_proc_cred-4814', 'mcc_proc_cred-4816', 'mcc_proc_cred-4829', 'mcc_proc_cred-4899', 'mcc_proc_cred-4900', 'mcc_proc_cred-5013', 'mcc_proc_cred-5039', 'mcc_proc_cred-5045', 'mcc_proc_cred-5047', 'mcc_proc_cred-5072', 'mcc_proc_cred-5111', 'mcc_proc_cred-5122', 'mcc_proc_cred-5192', 'mcc_proc_cred-5200', 'mcc_proc_cred-5211', 'mcc_proc_cred-5231', 'mcc_proc_cred-5251', 'mcc_proc_cred-5261', 'mcc_proc_cred-5309', 'mcc_proc_cred-5310', 'mcc_proc_cred-5311', 'mcc_proc_cred-5331', 'mcc_proc_cred-5399', 'mcc_proc_cred-5411', 'mcc_proc_cred-5422', 'mcc_proc_cred-5441', 'mcc_proc_cred-5451', 'mcc_proc_cred-5462', 'mcc_proc_cred-5499', 'mcc_proc_cred-5511', 'mcc_proc_cred-5533', 'mcc_proc_cred-5541', 'mcc_proc_cred-5542', 'mcc_proc_cred-5611', 'mcc_proc_cred-5621', 'mcc_proc_cred-5631', 'mcc_proc_cred-5641', 'mcc_proc_cred-5651', 'mcc_proc_cred-5655', 'mcc_proc_cred-5661', 'mcc_proc_cred-5691', 'mcc_proc_cred-5699', 'mcc_proc_cred-5712', 'mcc_proc_cred-5714', 'mcc_proc_cred-5719', 'mcc_proc_cred-5722', 'mcc_proc_cred-5732', 'mcc_proc_cred-5734', 'mcc_proc_cred-5735', 'mcc_proc_cred-5811', 'mcc_proc_cred-5812', 'mcc_proc_cred-5813', 'mcc_proc_cred-5814', 'mcc_proc_cred-5815', 'mcc_proc_cred-5816', 'mcc_proc_cred-5817', 'mcc_proc_cred-5818', 'mcc_proc_cred-5912', 'mcc_proc_cred-5921', 'mcc_proc_cred-5931', 'mcc_proc_cred-5941', 'mcc_proc_cred-5942', 'mcc_proc_cred-5943', 'mcc_proc_cred-5944', 'mcc_proc_cred-5945', 'mcc_proc_cred-5947', 'mcc_proc_cred-5948', 'mcc_proc_cred-5949', 'mcc_proc_cred-5964', 'mcc_proc_cred-5965', 'mcc_proc_cred-5967', 'mcc_proc_cred-5968', 'mcc_proc_cred-5969', 'mcc_proc_cred-5970', 'mcc_proc_cred-5973', 'mcc_proc_cred-5977', 'mcc_proc_cred-5983', 'mcc_proc_cred-5992', 'mcc_proc_cred-5993', 'mcc_proc_cred-5994', 'mcc_proc_cred-5995', 'mcc_proc_cred-5999', 'mcc_proc_cred-6010', 'mcc_proc_cred-6011', 'mcc_proc_cred-6051', 'mcc_proc_cred-6211', 'mcc_proc_cred-6300', 'mcc_proc_cred-6399', 'mcc_proc_cred-6513', 'mcc_proc_cred-6536', 'mcc_proc_cred-6538', 'mcc_proc_cred-6540', 'mcc_proc_cred-7011', 'mcc_proc_cred-7216', 'mcc_proc_cred-7230', 'mcc_proc_cred-7273', 'mcc_proc_cred-7278', 'mcc_proc_cred-7298', 'mcc_proc_cred-7299', 'mcc_proc_cred-7311', 'mcc_proc_cred-7333', 'mcc_proc_cred-7338', 'mcc_proc_cred-7372', 'mcc_proc_cred-7392', 'mcc_proc_cred-7399', 'mcc_proc_cred-7512', 'mcc_proc_cred-7519', 'mcc_proc_cred-7523', 'mcc_proc_cred-7538', 'mcc_proc_cred-7542', 'mcc_proc_cred-7622', 'mcc_proc_cred-7832', 'mcc_proc_cred-7841', 'mcc_proc_cred-7922', 'mcc_proc_cred-7941', 'mcc_proc_cred-7994', 'mcc_proc_cred-7995', 'mcc_proc_cred-7996', 'mcc_proc_cred-7997', 'mcc_proc_cred-7999', 'mcc_proc_cred-8011', 'mcc_proc_cred-8021', 'mcc_proc_cred-8050', 'mcc_proc_cred-8062', 'mcc_proc_cred-8071', 'mcc_proc_cred-8099', 'mcc_proc_cred-8220', 'mcc_proc_cred-8299', 'mcc_proc_cred-8351', 'mcc_proc_cred-8398', 'mcc_proc_cred-8641', 'mcc_proc_cred-8661', 'mcc_proc_cred-8999', 'mcc_proc_cred-9222', 'mcc_proc_cred-9311', 'mcc_proc_cred-9399', 'mcc_proc_cred-9402']
    mcc_proc_cred = mcc_proc_cred.reindex(columns=cols)
    mcc_proc_cred = mcc_proc_cred[cols]
    mcc_proc_cred = mcc_proc_cred.fillna(0)

    mcc_proc_cred_stats = pd.DataFrame()
    mcc_proc_cred_stats.index = mcc_proc_cred.index
    mcc_proc_cred_stats = get_stats(mcc_proc_cred_stats, 'mcc_proc_cred_stats')

    # ------------

    mcc_proc_deb['mcc_proc_deb_sum'] = debit_sum / (debit_sum + credit_sum)
    mcc_proc_cred['mcc_proc_cred_sum'] = credit_sum / (debit_sum + credit_sum)

    # ------------

    dm_freq = transactions.pivot_table(
        index='user_id', columns=['day_of_month'], values=['transaction_amt'], 
        aggfunc=['count'], fill_value=0)
    dm_freq.columns = [f'dm_freq-{i[2]}' for i in dm_freq.columns]

    cols = ['dm_freq-1', 'dm_freq-2', 'dm_freq-3', 'dm_freq-4', 'dm_freq-5', 'dm_freq-6', 'dm_freq-7', 'dm_freq-8', 'dm_freq-9', 'dm_freq-10', 'dm_freq-11', 'dm_freq-12', 'dm_freq-13', 'dm_freq-14', 'dm_freq-15', 'dm_freq-16', 'dm_freq-17', 'dm_freq-18', 'dm_freq-19', 'dm_freq-20', 'dm_freq-21', 'dm_freq-22', 'dm_freq-23', 'dm_freq-24', 'dm_freq-25', 'dm_freq-26', 'dm_freq-27', 'dm_freq-28', 'dm_freq-29', 'dm_freq-30', 'dm_freq-31']
    dm_freq = dm_freq.reindex(columns=cols)
    dm_freq = dm_freq[cols]
    dm_freq = dm_freq.fillna(0)

    dm_freq_stats = pd.DataFrame()
    dm_freq_stats.index = dm_freq.index
    dm_freq_stats['dm_freq_stats_sum'] = dm_freq.sum(axis=1)
    dm_freq_stats = get_stats(dm_freq_stats, 'dm_freq_stats')

    # ------------

    dm_freq_proc = dm_freq.div(dm_freq.sum(axis=1), axis=0).fillna(0) # процентный вид, сумма строки = 1
    dm_freq_proc.columns = [f'dm_freq_proc-{i}' for i in dm_freq_proc.columns]

    cols = ['dm_freq_proc-dm_freq-1', 'dm_freq_proc-dm_freq-2', 'dm_freq_proc-dm_freq-3', 'dm_freq_proc-dm_freq-4', 'dm_freq_proc-dm_freq-5', 'dm_freq_proc-dm_freq-6', 'dm_freq_proc-dm_freq-7', 'dm_freq_proc-dm_freq-8', 'dm_freq_proc-dm_freq-9', 'dm_freq_proc-dm_freq-10', 'dm_freq_proc-dm_freq-11', 'dm_freq_proc-dm_freq-12', 'dm_freq_proc-dm_freq-13', 'dm_freq_proc-dm_freq-14', 'dm_freq_proc-dm_freq-15', 'dm_freq_proc-dm_freq-16', 'dm_freq_proc-dm_freq-17', 'dm_freq_proc-dm_freq-18', 'dm_freq_proc-dm_freq-19', 'dm_freq_proc-dm_freq-20', 'dm_freq_proc-dm_freq-21', 'dm_freq_proc-dm_freq-22', 'dm_freq_proc-dm_freq-23', 'dm_freq_proc-dm_freq-24', 'dm_freq_proc-dm_freq-25', 'dm_freq_proc-dm_freq-26', 'dm_freq_proc-dm_freq-27', 'dm_freq_proc-dm_freq-28', 'dm_freq_proc-dm_freq-29', 'dm_freq_proc-dm_freq-30', 'dm_freq_proc-dm_freq-31']
    dm_freq_proc = dm_freq_proc.reindex(columns=cols)
    dm_freq_proc = dm_freq_proc[cols]
    dm_freq_proc = dm_freq_proc.fillna(0)

    dm_freq_proc_stats = pd.DataFrame()
    dm_freq_proc_stats.index = dm_freq.index
    dm_freq_proc_stats = get_stats(dm_freq_proc_stats, 'dm_freq_proc_stats')

    # ------------

    dw_freq = transactions.pivot_table(
        index='user_id', columns=['day_of_week'], values=['transaction_amt'], 
        aggfunc=['count'], fill_value=0)
    dw_freq.columns = [f'dw_freq-{i[2]}' for i in dw_freq.columns]

    cols = ['dw_freq-0', 'dw_freq-1', 'dw_freq-2', 'dw_freq-3', 'dw_freq-4', 'dw_freq-5', 'dw_freq-6']
    dw_freq = dw_freq.reindex(columns=cols)
    dw_freq = dw_freq[cols]
    dw_freq = dw_freq.fillna(0)

    dw_freq_stats = pd.DataFrame()
    dw_freq_stats.index = dw_freq.index
    dw_freq_stats['dw_freq_stats_sum'] = dw_freq.sum(axis=1)
    dw_freq_stats = get_stats(dw_freq_stats, 'dw_freq_stats')

    # ------------

    dw_freq_proc = dw_freq.div(dw_freq.sum(axis=1), axis=0).fillna(0) # процентный вид, сумма строки = 1
    dw_freq_proc.columns = [f'dw_freq_proc-{i}' for i in dw_freq_proc.columns]

    cols = ['dw_freq_proc-dw_freq-0', 'dw_freq_proc-dw_freq-1', 'dw_freq_proc-dw_freq-2', 'dw_freq_proc-dw_freq-3', 'dw_freq_proc-dw_freq-4', 'dw_freq_proc-dw_freq-5', 'dw_freq_proc-dw_freq-6']
    dw_freq_proc = dw_freq_proc.reindex(columns=cols)
    dw_freq_proc = dw_freq_proc[cols]
    dw_freq_proc = dw_freq_proc.fillna(0)

    dw_freq_proc_stats = pd.DataFrame()
    dw_freq_proc_stats.index = dw_freq.index
    dw_freq_proc_stats = get_stats(dw_freq_proc_stats, 'dw_freq_proc_stats')

    # ------------

    hd_freq = transactions.pivot_table(
        index='user_id', columns=['hour_of_day'], values=['transaction_amt'], 
        aggfunc=['count'], fill_value=0)
    hd_freq.columns = [f'hd_freq-{i[2]}' for i in hd_freq.columns]

    cols = ['hd_freq-0', 'hd_freq-1', 'hd_freq-2', 'hd_freq-3', 'hd_freq-4', 'hd_freq-5', 'hd_freq-6', 'hd_freq-7', 'hd_freq-8', 'hd_freq-9', 'hd_freq-10', 'hd_freq-11', 'hd_freq-12', 'hd_freq-13', 'hd_freq-14', 'hd_freq-15', 'hd_freq-16', 'hd_freq-17', 'hd_freq-18', 'hd_freq-19', 'hd_freq-20', 'hd_freq-21', 'hd_freq-22', 'hd_freq-23']
    hd_freq = hd_freq.reindex(columns=cols)
    hd_freq = hd_freq[cols]
    hd_freq = hd_freq.fillna(0)

    hd_freq_stats = pd.DataFrame()
    hd_freq_stats.index = hd_freq.index
    hd_freq_stats['hd_freq_stats_sum'] = hd_freq.sum(axis=1)
    hd_freq_stats = get_stats(hd_freq_stats, 'hd_freq_stats')

    # ------------

    hd_freq_proc = hd_freq.div(hd_freq.sum(axis=1), axis=0).fillna(0) # процентный вид, сумма строки = 1
    hd_freq_proc.columns = [f'hd_freq_proc-{i}' for i in hd_freq_proc.columns]

    cols = ['hd_freq_proc-hd_freq-0', 'hd_freq_proc-hd_freq-1', 'hd_freq_proc-hd_freq-2', 'hd_freq_proc-hd_freq-3', 'hd_freq_proc-hd_freq-4', 'hd_freq_proc-hd_freq-5', 'hd_freq_proc-hd_freq-6', 'hd_freq_proc-hd_freq-7', 'hd_freq_proc-hd_freq-8', 'hd_freq_proc-hd_freq-9', 'hd_freq_proc-hd_freq-10', 'hd_freq_proc-hd_freq-11', 'hd_freq_proc-hd_freq-12', 'hd_freq_proc-hd_freq-13', 'hd_freq_proc-hd_freq-14', 'hd_freq_proc-hd_freq-15', 'hd_freq_proc-hd_freq-16', 'hd_freq_proc-hd_freq-17', 'hd_freq_proc-hd_freq-18', 'hd_freq_proc-hd_freq-19', 'hd_freq_proc-hd_freq-20', 'hd_freq_proc-hd_freq-21', 'hd_freq_proc-hd_freq-22', 'hd_freq_proc-hd_freq-23']
    hd_freq_proc = hd_freq_proc.reindex(columns=cols)
    hd_freq_proc = hd_freq_proc[cols]
    hd_freq_proc = hd_freq_proc.fillna(0)

    hd_freq_proc_stats = pd.DataFrame()
    hd_freq_proc_stats.index = hd_freq.index
    hd_freq_proc_stats = get_stats(hd_freq_proc_stats, 'hd_freq_proc_stats')

    # ------------

    time_features = transactions.groupby('user_id')['time_day'].agg(['mean', 'std', 'min', 'max', 'median', 'sem', 'skew', 'var', 'sum'])
    time_features.columns = [f'tr_time_{c}' for c in time_features.columns]
    time_features['tr_time_amplitude'] = time_features['tr_time_max'] - time_features['tr_time_min']
    
    # ---------->
    
    
    df_test = pd.concat([
        result.set_index('user_id').target.rename('nn_predict'),
        time_features,
        
        mcc_freq,
        mcc_freq_stats,
        dm_freq,
        dm_freq_stats,
        dw_freq,
        dw_freq_stats,
        hd_freq,
        hd_freq_stats,

        mcc_freq_proc,
        dm_freq_proc,
        dw_freq_proc,
        hd_freq_proc,

        #mcc_proc,
        mcc_proc_deb,
        mcc_proc_cred,

        #mcc_proc_stats,
        #mcc_proc_deb_stats,
        #mcc_proc_cred_stats,
        
    ], axis=1)
    
    if len(X_train) >= batch_size:
        df_test = pd.concat([df_test, df_z], axis=1)
        df_test = df_test.dropna()
        model_cb = CatBoostClassifier().load_model("./model_dir/model_cb_vrae.cbm")
    else:
        model_cb = CatBoostClassifier().load_model("./model_dir/model_cb.cbm")
           
    columns = model_cb.get_feature_importance(prettified=True)['Feature Id'].values
    
    for col in columns:
        if col not in df_test.columns:
            df_test[col] = 0
    predicts = model_cb.predict_proba(df_test[columns])[:,1]
    
    sub_user_id = np.array(df_test.index)
    sub_predicts = np.array(predicts)
    
    for u in u_drop:
        sub_user_id = np.append(sub_user_id, u)
        sub_predicts = np.append(sub_predicts, np.max(sub_predicts))
        
    submission = pd.DataFrame({"user_id": sub_user_id, "target": sub_predicts})
    submission = submission.sort_values('user_id')
    submission.to_csv(output_path, index=False)
    


if __name__ == "__main__":
    main()