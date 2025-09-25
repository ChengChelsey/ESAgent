

from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances_argmin_min
import math
#from isotree import IsolationForest
# 生成示例数据
 
datasets = [ "AIOPS","TODS", "UCR","WSD"]
#datasets = ["WSD"]
method = "tree"
eps = 1e-15


base_folder_final = '/home/mjp/tsad/122/Results/Scores'
directory_names1 = ['AE', 'Donut', 'EncDecAD', 'LSTMADalpha', 'LSTMADbeta']
directory_names1 = ['AE', 'Donut','EncDecAD', 'LSTMADalpha', 'FCVAE','AR', 'DAMP', 'Diff']
aaabbvbv = ['AE', 'Donut','EncDecAD', 'LSTMADalpha', 'LSTMADbeta', 'FCVAE']

directory_names1 = ['Donut']

specific_path = 'one_by_one'

aaaaaaa = '/home/mjp/tsad/122/datasets/UTS'

for i in range(len(aaabbvbv)):
    directory_names1 = aaabbvbv[i]
    print('###########################################################3')
    print(directory_names1)

    for dataset in datasets:

        directory_names2 = []


        path1 = os.path.join(base_folder_final, 'AE', specific_path, dataset)

        if os.path.exists(path1):
            for file_name in os.listdir(path1):
                if file_name.endswith('.npy'):
                    directory_names2.append(file_name)

        #print(directory_names2)
                    
        p_set = []
        r_set = []
        for directory_name2 in directory_names2:


            search_set = []
            tot_anomaly = 0
            tot_anomalyall = 0
            key = 0
            dataall = np.array([])
            pathfinal2 = os.path.join(base_folder_final, method, specific_path, dataset, directory_name2)

            pathfinal1 = os.path.join(base_folder_final, 'Diff', specific_path, dataset, directory_name2)

            pathfinal1_test = os.path.join(base_folder_final, directory_names1, specific_path, dataset, directory_name2)
            labelaaaa = os.path.join(aaaaaaa, dataset, directory_name2[:-4],'test_label.npy')


            arrone = np.load(pathfinal1)

            scores = np.load(pathfinal1_test)

            labels = np.load(labelaaaa)



            cha = len(scores) - len(arrone)
            if cha > 0:
                scores = scores[cha:]
            else:
                arrone = arrone[-cha:]

            cha = len(labels) - len(arrone)

            labels = labels[cha:]


            for i in range(labels.shape[0]):
                tot_anomalyall += (labels[i] > 0.5)

            if tot_anomalyall == 0:
                p_set.append(1)
                continue


            scores_two = scores
            # print(len(scores),len(labels))
            scores = arrone

            #scores = np.insert(scores, 0, 0)

            flag = 0
            cur_anomaly_len = 0
            cur_max_anomaly_score = 0
            for i in range(labels.shape[0]):
                if labels[i] > 0.5:
                    # record the highest score in an anomaly segment
                    if flag == 1:
                        cur_anomaly_len += 1
                        cur_max_anomaly_score = scores[i] if scores[i] > cur_max_anomaly_score else cur_max_anomaly_score  # noqa: E501
                    else:
                        flag = 1
                        cur_anomaly_len = 1
                        cur_max_anomaly_score = scores[i]
                else:
                    # reconstruct the score using the highest score
                    if flag == 1:
                        flag = 0
                        search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
                        search_set.append((scores[i], 1, False))
                    else:
                        search_set.append((scores[i], 1, False))
            if flag == 1:
                search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
                

            #search_set.sort(key=lambda x: x[0], reverse=True)

            search_set_two = search_set.copy()


            scores = scores_two
            flag = 0
            cur_anomaly_len = 0
            cur_max_anomaly_score = 0
            search_set = []
            for i in range(labels.shape[0]):
                if labels[i] > 0.5:
                    # record the highest score in an anomaly segment
                    if flag == 1:
                        cur_anomaly_len += 1
                        cur_max_anomaly_score = scores[i] if scores[i] > cur_max_anomaly_score else cur_max_anomaly_score  # noqa: E501
                    else:
                        flag = 1
                        cur_anomaly_len = 1
                        cur_max_anomaly_score = scores[i]
                else:
                    # reconstruct the score using the highest score
                    if flag == 1:
                        flag = 0
                        search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
                        search_set.append((scores[i], 1, False))
                    else:
                        search_set.append((scores[i], 1, False))
            if flag == 1:
                search_set.append((cur_max_anomaly_score, cur_anomaly_len, True))
                
            #search_set.sort(key=lambda x: x[0], reverse=True)

            search_set_three = []

            # search_set_two_copy = search_set_two.copy()

            # search_set_two_copy.sort(key=lambda x: x[0], reverse=True)

            # x = math.ceil(len(search_set_two_copy) * 0.30)

            # x_zhi = search_set_two_copy[x][0]

            for i in range(len(search_set)):
                if search_set_two[i][0] > 0.5:
                    search_set_three.append(search_set[i])


            search_set_three.sort(key=lambda x: x[0], reverse=True)

            search_set= search_set_three

            for i in range(len(search_set)):
                if search_set[i][2]:  # for an anomaly point
                    tot_anomaly += search_set[i][1]

            tot_anomaly  = tot_anomalyall
            



            best_f1 = 0
            threshold = 0
            P = 0
            TP = 0
            best_P = 0
            best_TP = 0
            for i in range(len(search_set)):
                P += search_set[i][1]
                if search_set[i][2]:  # for an anomaly point
                    TP += search_set[i][1]
                precision = TP / (P + eps)
                recall = TP / (tot_anomaly + eps)
                f1 = 2 * precision * recall / (precision + recall + eps)
                if f1 > best_f1:
                    best_f1 = f1
                    threshold = search_set[i][0]
                    best_P = P
                    best_TP = TP
            precision = best_TP / (best_P + eps)
            recall = best_TP / (tot_anomaly + eps)


            p_set.append(best_f1)



        p = np.mean(p_set)
        print(p)
