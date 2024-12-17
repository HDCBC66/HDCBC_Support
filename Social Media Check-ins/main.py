import pickle
import os
import pandas as pd
import numpy as np
import time
import shutil
import seaborn as sns
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import cluster_acc
import HDCBC as HD

sns.set_context('poster')
sns.set_color_codes()


def clear_folder(folder_path):
    """清空指定文件夹"""
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def plot_clusters(data_name, data, algorithm, kwds,k):
    """
    执行 HDCBC 聚类并绘制结果
    :param data_name: 数据集名称
    :param data: 数据 (特征和标签)
    :param algorithm: 聚类算法 (HDCBC)
    :param kwds: 算法参数
    :return: 数据使用情况、ARS、NMI、ACC 和运行时间
    """
    labels_original = data.iloc[:, -1].values  # 原始标签
    
    print(labels_original)
    
    folder_path = f'result/{algorithm.__name__}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    try:
        start_time = time.time()
        
        
        attribute_list=['week_attr', 'hour']
        labels = algorithm(**kwds).fit_predict_with_attribute(data_name,data,attribute_list,k)
        
        print(labels)
            
        #labels = algorithm(**kwds).fit_predict(data_name, data)
        end_time = time.time()

        ars = adjusted_rand_score(labels_original, labels)
        nmi = normalized_mutual_info_score(labels_original, labels)
        acc = cluster_acc(labels_original, labels)
        time_use = end_time - start_time

    except Exception as e:
        print(e)
        ars, nmi, acc, time_use = 'Null', 'Null', 'Null', 'timeout'

    return ars, nmi, acc, time_use


if __name__ == "__main__":
    # 加载数据
    with open(r'cache/data_csv.pkl', 'rb') as file:
        data = pickle.load(file)

    results = []
    attr = []

    # 遍历数据集
    for key in data.keys():
        acc1 = 0
        att = {}
        result = {}

        # HDCBC 参数搜索空间
        list1 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        list2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        list3 = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2]
        list4 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 50]

        data_name = key
        for i in list1:
            for j in list2:
                for m in list3:
                    for n1 in list4:
                        
                        
                        parameter = {'K_DCM': i,
                                     'K_nearest': j,
                                     'CM_threshold': m,
                                     'minclustersize': n1}
                        '''
                        fixed_params = {
                        'K_DCM': 7,
                        'K_nearest': 5,
                        'CM_threshold': 0.2,
                        'minclustersize': 15
                        }
                        '''

                        ars, nmi, acc, time_use = plot_clusters(data_name, data[key], HD.HDCBC, parameter,k=2)
                        
                        print(ars)

                        if acc > acc1:
                            acc1 = acc
                            i1, j1, m1, n11 = i, j, m, n1

                    clear_folder(f'cache/{data_name}')

        # 保存最优参数
        att['K_DCM'] = i1
        att['K_nearest'] = j1
        att['CM_threshold'] = m1
        att['minclustersize'] = n11

        parameter = {'K_DCM': i1,
                     'K_nearest': j1,
                     'CM_threshold': m1,
                     'minclustersize': n11}

        # 重新评估最优参数下的结果
        ars, nmi, acc, time_use = plot_clusters(data_name, data[key], HD.HDCBC, parameter,k=2)
        result['HDCBC'] = [ars, nmi, acc, time_use]

        results.append([data_name, result])
        attr.append([data_name, att])

    # 生成最终结果文件
    ars_result = {}
    nmi_result = {}
    acc_result = {}
    time_use_result = {}

    for row in results:
        ars_result['data_name'] = []
        nmi_result['data_name'] = []
        acc_result['data_name'] = []
        time_use_result['data_name'] = []

        ars_result['data_name'].append(row[0])
        nmi_result['data_name'].append(row[0])
        acc_result['data_name'].append(row[0])
        time_use_result['data_name'].append(row[0])

        for key in row[1].keys():
            if key in ars_result:
                ars_result[key].append(row[1][key][0])
            else:
                ars_result[key] = [row[1][key][0]]

            if key in nmi_result:
                nmi_result[key].append(row[1][key][1])
            else:
                nmi_result[key] = [row[1][key][1]]

            if key in acc_result:
                acc_result[key].append(row[1][key][2])
            else:
                acc_result[key] = [row[1][key][2]]

            if key in time_use_result:
                time_use_result[key].append(row[1][key][3])
            else:
                time_use_result[key] = [row[1][key][3]]

    # 保存为 CSV 文件
    df_ars = pd.DataFrame(ars_result)
    df_nmi = pd.DataFrame(nmi_result)
    df_acc = pd.DataFrame(acc_result)
    df_time_use = pd.DataFrame(time_use_result)
    att_out = pd.DataFrame(attr)

    att_out.to_csv('att.csv')
    df_ars.to_csv('ars.csv')
    df_nmi.to_csv('nmi.csv')
    df_acc.to_csv('acc.csv')
    df_time_use.to_csv('time_use.csv')
