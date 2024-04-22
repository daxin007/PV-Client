import os 
import argparse
import random 
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler
import warnings 
warnings.simplefilter('ignore')

from models.Client import Model

def getRes(args, data_input, scaler_,k, device):
    configs = argparse.ArgumentParser()
    configs.pred_len = k
    configs.output_attention = False
    configs.seq_len = args.seq_len
    configs.n_heads = 4
    configs.dropout = 0.1
    configs.d_ff = 128
    configs.activation = 'gelu'
    configs.factor = 3
    configs.e_layers = 2
    configs.w_lin = 1.0
    configs.w_trans = 1.0
    # configs.enc_in = 25
    configs.enc_in = 6
    model = Model(configs)
    model.load_state_dict(torch.load('model/model.pt'))
    model.to(device)
    model.eval()

    # mean_ = scaler_.mean_[-1]
    # scale_ = scaler_.scale_[-1]
    
    # data
    data = data_input.copy()
    
    features2 = ['if30', 'if90', 'if100', 'if70', 'if110','ir10']
    data_s = data[features2].values
    data_scaled = scaler_.fit_transform(data_s)

    df_input = pd.DataFrame(data = data_scaled, columns = ['if30', 'if90', 'if100', 'if70', 'if110','ir10'])
    df_input['time'] = data['time']

    df_input['ir10_k'] = df_input['ir10'].shift(k)

    df_input.dropna(inplace = True)

    df_input_s = df_input[['if30', 'if90', 'if100', 'if70', 'if110', 'ir10_k', 'ir10']]

    df_x = df_input_s.values
    mean_ = scaler_.mean_[-1]
    scale_ = scaler_.scale_[-1]
    y_predList = []

    with torch.no_grad():
        for i in range(0, len(df_x) - args.seq_len + 1-k, k):
            x_i = df_x[i: i + args.seq_len, 0:6]
            y_i = df_x[i + args.seq_len: i + args.seq_len+k, 6]

            x_i = torch.tensor(x_i)
            x_i = x_i.float().to(device)
            x_i = x_i.unsqueeze(0)
            # print(x_i.shape)
            # print(y_i.shape)
            # print(model)
            output_ = model(x_i, None, None, None)[0,:,-1] + model(x_i, None, None, None)[0,:,0]
            y_pred = output_*scale_+mean_
            y_predList.extend(list(y_pred.cpu().numpy()))
    return y_predList

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type = int, default = 2)
    parser.add_argument('--hid_dim', type = int, default = 6)

    parser.add_argument('--seq_len', type = int, default = 192)

    parser.add_argument('--seed', type = int, default = 0)

    parser.add_argument('--dataPath', type = str, default = './input')

    parser.add_argument('--fileNameR', type = str, default = 'input_real.xlsx')
    parser.add_argument('--fileNameF', type = str, default = 'input_forecast.xlsx')
    parser.add_argument('--fileNameS', type = str, default = 'input_station.xlsx')

    parser.add_argument('--modelPath', type = str, default = './model')

    parser.add_argument('--outPath', type = str, default = './output')
    parser.add_argument('--outfile', type = str, default = 'output_result.xlsx')

    parser.add_argument('--scalerPath', type = str, default = './scaler')
    parser.add_argument('--scalerName', type = str, default = 'scaler.save')
    parser.add_argument('--cap', type=int, default=375)

    args = parser.parse_args()

    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    scaler_ = StandardScaler()

    # scaler_ = joblib.load(os.path.join(args.scalerPath, args.scalerName))
    #print(data_real.shape)

    #data loading

    filePathF = os.path.join(args.dataPath, args.fileNameF)
    df_f = pd.read_excel(filePathF)
    featuresF = ['time', 'if30']
    df_fs = df_f[featuresF]

    filePathR = os.path.join(args.dataPath, args.fileNameR)
    df_r = pd.read_excel(filePathR)
    featuresR = ['time', 'ir10']
    df_rs = df_r[featuresR]

    df_m = pd.merge(df_f, df_r)

    #station information
    
    filePathS = os.path.join(args.dataPath, args.fileNameS)
    df_s = pd.read_excel(filePathS)
    featuresS = ['time', 'is10', 'is20']
    df_ss = df_s[featuresS]

    df_ss['coef'] = df_ss['is20']/df_ss['is10']
    df_ss_  = df_ss[args.seq_len+96:]
    y_res = getRes(args, df_m, scaler_,96, device)
    # print(len(y_res))

    df_o = pd.DataFrame(data=y_res, columns=['power'])
    # print(len(df_o))

    df_f['time'] = pd.to_datetime(df_f['time'], format='%Y-%m-%d %H:%M:%S')
    df_f = df_f[args.seq_len+96:]
    df_f.set_index(['time'], inplace=True)
    df_o['time'] = df_f.index

    df_o['if30'] = df_f['if30'].values

    df_o.loc[df_o['if30'] <= 0, 'power'] = 0
    df_o.loc[df_o['power'] <= 0, 'power'] = 0
    df_o.loc[df_o['power'] >= args.cap, 'power'] = args.cap

    df_o['or10'] = df_o['power'] * df_ss_['coef'].to_numpy()

    df_o['or20'] = df_o['or10'] * 1.2
    df_o['or30'] = df_o['or10'] * 0.8

    features = ['time', 'or10', 'or20', 'or30']
    df_oo = df_o[features]
    df_oo.to_excel(os.path.join(args.outPath, args.outfile), float_format = '%.3f', index = False)



if __name__ == '__main__':
    main()
