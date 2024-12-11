# 此脚本需要优化，输出的csv格式不利于读取和作图
# Python import
import os
import copy
import random
import collections
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import svm
import warnings
import joblib
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# %%
#Evaluate performance of model
def evaluate_performance(y_test, y_pred, y_prob):
    # AUROC
    auroc = metrics.roc_auc_score(y_test,y_prob)
    auroc_curve = metrics.roc_curve(y_test, y_prob)
    # AUPRC
    auprc=metrics.average_precision_score(y_test, y_prob) 
    auprc_curve=metrics.precision_recall_curve(y_test, y_prob)
    #Accuracy
    accuracy=metrics.accuracy_score(y_test,y_pred) 
    #MCC
    mcc=metrics.matthews_corrcoef(y_test,y_pred)
    
    recall=metrics.recall_score(y_test, y_pred)
    precision=metrics.precision_score(y_test, y_pred)
    f1=metrics.f1_score(y_test, y_pred)
    class_report=metrics.classification_report(y_test, y_pred,target_names = ["control","case"])

    model_perf = {"auroc":auroc,"auroc_curve":auroc_curve,
                  "auprc":auprc,"auprc_curve":auprc_curve,
                  "accuracy":accuracy, "mcc": mcc,
                  "recall":recall,"precision":precision,"f1":f1,
                  "class_report":class_report}
        
    return model_perf

# %%
# Output result of evaluation
def eval_output(model_perf,path):
    with open(os.path.join(path,"Evaluate_Result_TestSet.txt"),'w') as f:
        f.write("AUROC=%s\tAUPRC=%s\tAccuracy=%s\tMCC=%s\tRecall=%s\tPrecision=%s\tf1_score=%s\n" %
               (model_perf["auroc"],model_perf["auprc"],model_perf["accuracy"],model_perf["mcc"],model_perf["recall"],model_perf["precision"],model_perf["f1"]))
        f.write("\n######NOTE#######\n")
        f.write("#According to help_documentation of sklearn.metrics.classification_report:in binary classification, recall of the positive class is also known as sensitivity; recall of the negative class is specificity#\n\n")
        f.write(model_perf["class_report"])

# %%
# Plot AUROC of model
def plot_AUROC(model_perf,path):
    #get AUROC,FPR,TPR and threshold
    roc_auc = model_perf["auroc"]
    fpr,tpr,threshold = model_perf["auroc_curve"]
    #return AUROC info
    temp_df = pd.DataFrame({"FPR":fpr,"TPR":tpr})
    temp_df.to_csv(os.path.join(path,"AUROC_info.txt"),header = True,index = False, sep = '\t')
    #plot
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC (area = %0.2f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUROC of Models")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path,"AUROC_TestSet.pdf"),format = "pdf")

# %%
# Random seed
SEED = 100
random.seed(SEED)
np.random.seed(SEED)

warnings.filterwarnings(action='ignore')

# Output dir
output_dir = "./ML_Model_Output"
if not (os.path.exists(output_dir)):
    os.mkdir(output_dir)

# %%
# 合并所有样本
folder_path = '/BioII/lulab_b/huangkeyun/zhangys/alkb-seq/resources/NomalSamples/labels/'

# 获取该文件夹下所有 CSV 文件的文件名，并按文件名顺序排序
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

# 创建一个空的列表，用于存储所有的 DataFrame
dfs = []

# 依次读取每个 CSV 文件并将其添加到列表中
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)  # 读取 CSV 文件
    dfs.append(df)  # 将 DataFrame 添加到列表中

# 将所有的 DataFrame 按列合并
labels = pd.concat(dfs, axis=0)

# 设置文件夹路径
folder_path = '/BioII/lulab_b/huangkeyun/zhangys/alkb-seq/resources/NomalSamples/samples/'
# 获取该文件夹下所有 CSV 文件的文件名，并按文件名顺序排序
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
# 创建一个空的列表，用于存储所有的 DataFrame
dfs = []
# 依次读取每个 CSV 文件并将其添加到列表中
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)  # 读取 CSV 文件
    dfs.append(df)  # 将 DataFrame 添加到列表中
# 将所有的 DataFrame 按列合并
dataset = pd.concat(dfs, axis=0)


# %%
# 加载两个数据集，对样本数较多的数据集进行下采样平衡
#dataset = pd.read_csv('/BioII/lulab_b/huangkeyun/zhangys/alkb-seq/resources/SRR11004118_sample_prepared.csv')    
#labels = pd.read_csv('/BioII/lulab_b/huangkeyun/zhangys/alkb-seq/resources/SRR11004118_labels.csv')
dataset = pd.concat([dataset, labels['label']], axis=1)
df_y0 = dataset[dataset['label'] == 0]
df_y1 = dataset[dataset['label'] == 1]

# 确定两个子集中数量较少的那个
min_count = min(len(df_y0), len(df_y1))

# 从两个子集中随机选择等量的样本
df_y0_balanced = df_y0.sample(n=min_count, random_state=42) if len(df_y0) > min_count else df_y0
df_y1_balanced = df_y1.sample(n=min_count, random_state=42) if len(df_y1) > min_count else df_y1
# 合并这两个平衡后的子集
balanced_df = pd.concat([df_y0_balanced, df_y1_balanced])
# 打乱合并后的数据集的顺序
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
output_dir = './5fold_features_ablation'

# X = balanced_df.iloc[:, 1:-1].values
# y = balanced_df.iloc[:, -1].values
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# %%
def LR_kftest(X, y, output_dir = './5fold_features_ablation'):    
    # 创建五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # 存储每折的AUC
    auc_scores = []

    # Logistic Regression params
    lr_param_dict = {
        "penalty":["l2"],
        "C":[1e-3, 5e-3, 1e-2, 0.05, 0.1, 0.5,1,5,10,50,100,500,1000],
        "solver":["liblinear"],
        "random_state":[SEED]
    }

    # Initiate Logistic Regression model
    lr_model = LogisticRegression()

    # 存储每次训练的结果
    results = []

    # Five-fold Cross-validation
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}")
        
        # 获取当前折的训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Adjust hyper-parameters with randomized search
        lr_rscv = RandomizedSearchCV(lr_model, lr_param_dict, n_iter=100, cv=5, verbose=0,
                                    scoring="roc_auc", random_state=SEED, n_jobs=-1)
        lr_rscv.fit(X_train, y_train)

        # 获取当前折的AUC分数
        y_pred_proba = lr_rscv.best_estimator_.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_scores.append(auc)
        
        # 输出当前折的结果
        results.append({
            'fold': fold + 1,
            'AUC': auc
        })

    # 计算平均AUC和AUC方差
    mean_auc = np.mean(auc_scores)
    auc_variance = np.var(auc_scores)

    # # 打印五折的AUC、平均AUC和AUC方差
    # print(f"5-fold AUCs: {auc_scores}")
    # print(f"Mean AUC: {mean_auc}")
    # print(f"AUC Variance: {auc_variance}")

    # 将AUC及统计信息追加到输出结果文件中
    results.append({'fold': 'Mean AUC', 'AUC': mean_auc})
    results.append({'fold': 'AUC Variance', 'AUC': auc_variance})
    results_df = pd.DataFrame(results)
    # results_df['mean_AUC'] = mean_auc
    # results_df['AUC_variance'] = auc_variance

    # 保存结果
    path = os.path.join(output_dir, "LogisticRegression")
    os.makedirs(path, exist_ok=True)

    results_file_path = os.path.join(path, "LogisticRegression_AUCs.csv")

    # 判断文件是否已经存在
    if os.path.exists(results_file_path):
        # 如果文件已经存在，则以追加模式写入数据（不写列名）
        results_df.to_csv(results_file_path, mode='a', header=False, index=False)
    else:
        # 如果文件不存在，则创建新文件并写入列名
        results_df.to_csv(results_file_path, mode='w', header=True, index=False)

def run_lr_ablation_experiment(balanced_df, start, end, description, output_dir='./5fold_features_ablation'):
    """  
    参数：
    - balanced_df: 数据集的 DataFrame 对象。
    - start: int, 特征的起始列索引。
    - end: int, 特征的结束列索引。
    - description: str, 描述当前实验的说明，用于打印日志。
    - output_dir: str, 实验结果保存目录。
    """
    print(f"Running LR ablation experiment: {description}")
    X = balanced_df.iloc[:, start:end].values
    y = balanced_df.iloc[:, -1].values
    LR_kftest(X, y, output_dir=output_dir)

# 定义实验参数
lr_experiments = [
    {"start": 1, "end": -1, "description": "使用所有特征"},
    {"start": 1, "end": -1-160, "description": "消除onehot序列"},
    {"start": 1, "end": -1-200, "description": "消除onehot序列和第8个mod"},
    {"start": 1, "end": -1-240, "description": "消除onehot序列和第78个mod"},
    {"start": 1, "end": -1-280, "description": "消除onehot序列和第678个mod"},
    {"start": 1, "end": -1-320, "description": "消除onehot序列和第5678个mod"},
    {"start": 1, "end": -1-360, "description": "消除onehot序列和第45678个mod"},
    {"start": 1, "end": -1-400, "description": "消除onehot序列和第345678个mod"},
    {"start": 1, "end": -1-440, "description": "消除onehot序列和第2345678个mod"},
    {"start": 1, "end": -1-480, "description": "消除onehot序列和第12345678个mod"},
    {"start": 361, "end": -1, "description": "仅使用onehot序列"},
    {"start": 41, "end": 361, "description": "仅使用所有mod"},
    {"start": 41, "end": -1, "description": "使用所有mod和onehot"},
]

for exp in lr_experiments:
    run_lr_ablation_experiment(
        balanced_df,
        start=exp["start"],
        end=exp["end"],
        description=exp["description"]
    )


