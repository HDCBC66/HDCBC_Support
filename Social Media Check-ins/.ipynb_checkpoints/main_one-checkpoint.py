import pickle
import os
import pandas as pd
import geopandas
import math
import numpy as np
import copy
import time
import seaborn as sns

import sklearn
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import hdbscan
from sklearn.mixture import GaussianMixture
import HDCBC as HD
import sklearn.cluster as cluster
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import cluster_acc
#import CDC

sns.set_context('poster')
sns.set_color_codes()

def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def timeout_handler(signum, frame):
    print("time out")
    raise Exception("time out")

def plot_clusters(data_name,data, algorithm, args, kwds):
    print(algorithm.__name__)
    
    
    
    
    
    labels_original=data.iloc[:, -1].values

    folder_path1 = r'cache/' + data_name
    if not os.path.exists(folder_path1):
        os.mkdir(folder_path1)

    folder_path2 = r'result/'+algorithm.__name__
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)


    #超时跳出
    try:
        start_time = time.time()

        if algorithm.__name__!='HDCBC':
            if algorithm.__name__ == 'CDC':
                labels = algorithm(**kwds).fit_predict(data[:,:-1])
            else:
                labels = algorithm(*args, **kwds).fit_predict(data[:,:-1])

        else:

            k=2
            attribute_list=['week_attr', 'hour']
            labels = algorithm(**kwds).fit_predict_with_attribute(data_name,data,attribute_list,k)
            
        

        end_time = time.time()

        ars = adjusted_rand_score(labels_original, labels)
        
        
        
        nmi = normalized_mutual_info_score(labels_original, labels)
        acc = cluster_acc(labels_original, labels)
        time_use = end_time - start_time

        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plot_kwds = {'alpha': 1, 's': 1, 'linewidths': 0}

        fig, ax = plt.subplots()

        plt.scatter(data.iloc[:, 0].values, data.iloc[:, 1].values, c=colors, **plot_kwds)

        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)




    except Exception as e:
        print(e)
        ars='Null'
        nmi='Null'
        acc='Null'
        time_use='timeout'

        fig, ax = plt.subplots()
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
        text = f'ars:{ars}' + f',nmi:{ars}' + f',acc:{ars}' + f',time:{time_use}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, color='black', fontsize=12)


#Comparison of different Algorithms
    if algorithm.__name__ == 'HDCBC':
        plt.savefig(folder_path2 + r'/' + data_name + '_' + algorithm.__name__ + '.png', dpi=1000)
        plt.close(fig)
        return 1,ars, nmi, acc,time_use,labels

    else:
        plt.savefig(folder_path2 + r'/' + data_name + '_' + algorithm.__name__ + '.png', dpi=1000)
        plt.close(fig)

        return 1,ars, nmi, acc,time_use,labels




if __name__ == "__main__":
    
    # 固定超参数
    fixed_params = {
        'K_DCM': 7,
        'K_nearest': 5,
        'CM_threshold': 0.2,
        'minclustersize': 15
    }
    # 加载数据
    with open(r'cache/data_csv.pkl', 'rb') as file:
        data = pickle.load(file)

    results = []

    for key in data.keys():
        data_name = key
        
        # 使用固定超参数运行 HDCBC
        data_usage, ars, nmi, acc, time_use,_ = plot_clusters(
            data_name, data[key], HD.HDCBC, (), fixed_params
        )
        
        print('ars: '+str(ars))
        print('nmi: '+str(nmi))
        print('acc: '+str(acc))

        # 保存结果
        result = {
            'data_name': data_name,
            'ARS': ars,
            'NMI': nmi,
            'ACC': acc,
            'Time': time_use
        }
        results.append(result)

        # 清理缓存
        clear_folder('cache/' + data_name)

    # 保存结果到 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('HDCBC_results_one.csv', index=False)