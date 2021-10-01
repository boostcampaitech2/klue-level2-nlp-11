import pandas as pd
import numpy as np
from load_data import *
import random
open_path = "../dataset/train/train.csv"
output_path = "../dataset/train/train_new.csv"

def strtodic(s):
    '''
    json모듈의 load와 비슷한 역할을 함.
    dictionary 형태의 str을 받아서 dictionary로 바꿔줌.
    key, value로 string과 int만 인식 가능.
    '''
    rt = {}
    wd = False
    key = ''
    finished = ''
    temp = ''
    kvbool = False
    num = 0
    for i in s:
        if i == wd:
            wd = False
            finished = temp
            temp = ''
        elif wd == False:
            if (i == '"' or i == "'"):
                wd = i
            elif i == ',':
                if finished == '':
                    rt[key] = num
                    num = 0
                else:
                    rt[key] = finished
            elif i == ':' :
                key = finished
                finished = ''
            elif i.isdigit():
                num *= 10
                num += int(i)
        elif wd != i:
            temp += i
            
    rt[key] = finished
    return rt

def get_dict(Load_dataset):
    '''
    path의 train.csv파일을 분석.
    subject와 object의 type에 따라서 dictionary 반환.
    라벨별로 ind_list를 구하는 dictionary 반환.
    2중사전. ORG, PER ,DAT ,LOC이 sub/obj에 있는가에 따라 0->없음, 1->sub만, 2->obj만, 3-> 둘 다. 
    POH 또는 NOH를 포함하는 sub/obj는 변경하지 않음.
    '''
    tpdic = {}
    length = len(Load_dataset)
    lb_ind_dic = {}
    for i in range(length):
        data = Load_dataset.loc[i]
        lb = data['label']
        sbj = strtodic(data['subject_entity'].strip('{} '))
        try:
            tpdic[sbj['type']].add(sbj['word'])
        except:
            tpdic[sbj['type']] = set()
            tpdic[sbj['type']].add(sbj['word'])
        obj = strtodic(data['object_entity'].strip('{} '))
        try:
            tpdic[obj['type']].add(obj['word'])
        except:

            tpdic[obj['type']] = set()
            tpdic[obj['type']].add(obj['word'])
        dicvar = 0
        if sbj['type'] != 'POH' and sbj['type'] != 'NOH':
            dicvar += 1
        if obj['type'] != 'POH' and sbj['type'] != 'NOH':
            dicvar += 2
            
        try:
            lb_ind_dic[lb][dicvar].append(i)
        except:
            try:
                lb_ind_dic[lb][dicvar] = [i]
            except:
                lb_ind_dic[lb] = [[],[],[],[]]
                lb_ind_dic[lb][dicvar] = [i]
                

    for tp in tpdic:
        tpdic[tp] = list(tpdic[tp])
    return tpdic, lb_ind_dic

def make_data_ex(Load_dataset,tpdic,ind_list, option, N):
    '''
    INPUT
    dataset,dic: type->list-of-type(str), dic: label -> list of index
    option : 1 -> sub만 변경, 2-> obj만 변경, 3-> 둘 다 변경.
    N : 생성 숫자
    OUTPUT
    dataset에서 Object와 Subject를 동일 type로 바꾼 sentence.
    '''
    i = random.randint(0,len(ind_list)-1)
    while N>0:
        N-=1
        i+=1
        i%= len(ind_list)
        
        data = Load_dataset.loc[ind_list[i]]
        sentence = data['sentence']
        obj = strtodic(data['object_entity'].strip('{} '))
        sbj = strtodic(data['subject_entity'].strip('{} '))
        
        if option & 2 == 2:
            
            target_obj = tpdic[obj['type']][random.randint(0,len(tpdic[obj['type']])-1)]
            sentence = sentence[:obj['start_idx']] + target_obj +sentence[obj['end_idx']+1:]
        else:
            target_obj = obj['word']
            
        if option&1 == 1:
            
            target_sbj = tpdic[sbj['type']][random.randint(0,len(tpdic[sbj['type']])-1)]
            sentence = sentence[:sbj['start_idx']] + target_sbj +sentence[sbj['end_idx']+1:]
        else:
            target_sbj = sbj['word']
        
        yield [ str(target_sbj), str(target_obj), str(data['label']),sentence]

def create_data_csv(goal_num):  
    '''
    입력한 숫자 만큼 데이터를 생성합니다.
    '''
    Load_dataset = pd.read_csv(open_path)
    typedic, label_index_option_dic = get_dict(Load_dataset)
    f = open(output_path, 'w')
    f.write('id,subject_entity,object_entity,label,sentence\n')
    datanum = 0
    for lb in label_index_option_dic:
        print('====== MAKING '+lb)
        length_list = [len(label_index_option_dic[lb][i])for i in range(4)]
        dim1 = length_list[1]+length_list[2]
        dim2 = length_list[3]
        N = 0
        r=0
        while N < goal_num:
            r+=1
            N = r * dim1 + r * r * dim2
        if length_list[1]:
            for datalistform in make_data_ex(
                Load_dataset,typedic,label_index_option_dic[lb][1],
                1, r*len(label_index_option_dic[lb][1])):
                f.write(str(datanum)+',"')
                f.write('","'.join(datalistform)+'"\n')
                datanum += 1
        if length_list[2]:
            for datalistform in make_data_ex(
                Load_dataset,typedic,label_index_option_dic[lb][2],
                2, r*len(label_index_option_dic[lb][2])):
                f.write(str(datanum)+',"')
                f.write('","'.join(datalistform)+'"\n')
                datanum += 1
        if length_list[3]:
            for datalistform in make_data_ex(
                Load_dataset,typedic,label_index_option_dic[lb][3],
                3, goal_num - r * dim1):
                f.write(str(datanum)+',"')
                f.write('","'.join(datalistform)+'"\n')
                datanum += 1
    f.close()
    print('done')

if __name__ == '__main__':
    random.seed(212)
    create_data_csv(10000)