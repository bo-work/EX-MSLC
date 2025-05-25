from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def class_weight():
    # 标签0-5的样本数量
    # ids18
    class_counts = np.array([83145, 12533, 40486, 12848, 99709, 22584])

    classes = np.arange(6)  # 标签0-5

    # 创建完整的标签数组（每个标签重复出现对应次数）
    all_labels = []
    for i, count in enumerate(class_counts):
        all_labels.extend([i] * count)
    all_labels = np.array(all_labels)

    # 计算平衡权重
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=all_labels
    )
    # 输出结果
    for i, (count, weight) in enumerate(zip(class_counts, weights)):
        print(f"标签 {i}: 样本数 {count}, 权重 {weight:.4f}")


def heatmap_drag(df_cm):
    categories = ['Benign', 'Bot', 'BruteForce', 'DoS Hulk', 'HTTP DDoS', 'Infilteration']

    # for i, j in zip(y_gt, y_noi):
    #     y_gt_name.append(categories[i])
    #     y_noi_name.append(categories[j])
    # y_gt_name = np.array(y_gt_name)
    # y_noi_name = np.array(y_noi_name)
    # df_cm = pd.crosstab(y_gt_name, y_noi_name,
    #                     rownames=['Actual'], colnames=['Predicted'],
    #                     dropna=False, margins=False, normalize='index').round(4)
    # print(df_cm.to_dict())


    # ask1  'BruteForce': {'Benign': 0.0, 'Bot': 0, 'BruteForce': 0.0, 'DoS Hulk': 0., 'HTTP DDoS': 0.0, 'Infilteration': 0}
    # ==========
    # ask3  'Infilteration': {'Benign': 0.0, 'Bot': 0, 'BruteForce': 0.0, 'DoS Hulk': 0., 'HTTP DDoS': 0.0, 'Infilteration': 0}

    df_cm = pd.DataFrame()



    df_cm = df_cm[categories]
    df_cm = df_cm.reindex(categories)
    df_cm = df_cm.round(4)

    plt.figure(figsize=(16, 10))
    sns.set(font_scale=2)
    sns.heatmap(df_cm, cmap="viridis", annot=True, annot_kws={"size": 16})
    # sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    # 设置x、y轴标签为类别名称
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    # plt.xticks([])
    ####解决保存图时坐标文字显示不全#######
    plt.ylabel('')
    plt.xlabel('')
    plt.tight_layout()
    png_path = './results/imgs/ori.png'
    plt.show()

#
# class_weight()