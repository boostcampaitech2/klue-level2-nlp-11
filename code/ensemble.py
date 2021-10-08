import pickle as pickle
import numpy as np
import pandas as pd
import os
import argparse

def probs_to_float_array(probs):
    probs = probs.replace('[','')
    probs = probs.replace(']','')
    probs_list = probs.split(',')
    float_array = np.array(probs_list, dtype=float)
    
    return float_array

def probs_to_preds(probs):
    return probs.argmax()

def array_to_list(probs):
    return probs.tolist()

def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label



def ensemble(args):
    
    csv = args.csv_name.split(',')
    csv_list = []
    for name in csv:
        csv_list.append(pd.read_csv(os.path.join(args.csv_dir, name)))

    # apply float to array
    for i in range(len(csv_list)):
        csv_list[i].probs = csv_list[i].probs.apply(probs_to_float_array)

    # probability summation
    sum_probs = csv_list[0].probs
    for i in range(1, len(csv_list)):
        probs = csv_list[i].probs
        sum_probs += probs

    norm_sum_probs = sum_probs/ len(csv_list)

    preds = norm_sum_probs.apply(probs_to_preds)
    label_preds = num_to_label(preds)

    # Ai stages format
    probs_list = norm_sum_probs.apply(array_to_list)

    ensemble_csv = pd.DataFrame({'id':csv_list[0].id, 'pred_label':label_preds, 'probs':probs_list})
    ensemble_csv.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--csv_name', type=str, default='output1.csv,output2.csv,output3.csv,output4.csv', help='name of csv files. , is placed in between names')
    parser.add_argument('--csv_dir', type=str, default='./prediction', help='directory of files')
    parser.add_argument('--save_path', type=str, default='./prediction/ensemble.csv', help='directory of files')

    args = parser.parse_args()

    ensemble(args)

