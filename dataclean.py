import numpy as np
from scipy.special import softmax
from collections import Counter

# 提供每轮清洗的扩展子集

class Data_Clean():
    def __init__(self, dataset, max_size, delay_exp = 10, replace = True, strategy_type = '',  delay = 5):
        self.strategy_type = strategy_type
        self.dataset = dataset
        self.max_size = max_size
        self.allow_replace = replace
        self.data = self.dataset.train_data_fea
        self.label = self.dataset.train_labels
        self.label_true = self.dataset.train_labels_true
        self.clean_index = np.array([])
        self.clean_prob = np.array([])
        self.counter = 0
        self.delay_exp = delay_exp
        self.deley_size = int(self.max_size / self.delay_exp) + 1   #每次多提供的数量

        if self.strategy_type == 'sf':
            self.strategy = self.Self_Filtering(len(self.data), self.label, delay)



    def select_catetop_indices(self, list_label, list_index, list_prob, cate_size):
        # 存储每个标签对应的样本序列号和概率
        label_index_prob_dict = {}
        for label, index, prob in zip(list_label, list_index, list_prob):
            if label not in label_index_prob_dict:
                label_index_prob_dict[label] = []
            label_index_prob_dict[label].append((index, prob))

        # 对每个标签对应的样本按照概率排序并选取前 0.1 的样本
        selected_indices = []
        selected_probs = []
        for label, index_probs in label_index_prob_dict.items():
            sorted_index_probs = sorted(index_probs, key=lambda item: item[1], reverse=True)
            # num_top = max(1, int(len(sorted_index_probs) * rate))
            top_index_probs = sorted_index_probs[:cate_size]
            for index, prob in top_index_probs:
                selected_indices.append(index)
                selected_probs.append(prob)

        return selected_indices, selected_probs

    def get_clean_data(self, prec):
        # _, new_num = self.get_clean_index(prec)
        # if len(self.clean_index) == 0:
        #     return [],[],[]
        # return self.data[self.clean_index], self.label[self.clean_index], self.label_true[self.clean_index]
        self.get_clean_index(prec)
        if len(self.clean_index) == 0:
            return [],[],[]
        return self.data[self.clean_index], self.label[self.clean_index], self.label_true[self.clean_index]

    def get_clean_index(self, prec):
        old_len = len(self.clean_index)
        new_num = 0

        if (not self.allow_replace)  and  (old_len >= self.max_size):
            self.static_clean(old_len)
            return self.clean_index, new_num

        clean_index, clean_prob = self.strategy.get_clean_index(prec)

        if len(clean_index) > 0:
            self.counter += 1

            new_in_old_index = np.isin(clean_index, self.clean_index)
            new_index = ~new_in_old_index
            clean_index_new = clean_index[new_index]
            clean_prob_new = clean_prob[new_index]

            if self.allow_replace:
                # clean中的值可替换
                new_num = len(clean_index_new)
                self.clean_index = np.concatenate((self.clean_index, clean_index_new)).astype(np.int)
                self.clean_prob = np.concatenate((self.clean_prob, clean_prob_new))

                # 取概率较高的序号
                # sort_indices = np.argsort(-self.clean_prob)[:(self.deley_size * min(self.counter, self.delay_exp))]

                # 取每个类别中概率最高的序号
                clean_label = self.label[self.clean_index]
                clean_label_cate = len(set(clean_label))
                cate_size = int(self.deley_size * min(self.counter, self.delay_exp) / clean_label_cate)
                sort_indices, sort_probs = self.select_catetop_indices(clean_label, self.clean_index, self.clean_prob, cate_size)

                # self.clean_index = self.clean_index[sort_indices].astype(np.int)
                # self.clean_prob = self.clean_prob[sort_indices]
                self.clean_index = np.array(sort_indices, dtype = np.int64)
                self.clean_prob = np.array(sort_probs)

            else:
                # clean中的值不进行替换，只添加
                # allow_size = ((self.deley_size * min(self.counter, self.delay_exp))) - len(self.clean_index)
                sort_indices = np.argsort(-clean_prob_new)[:self.deley_size ]
                self.clean_index = np.concatenate((self.clean_index, clean_index_new[sort_indices])).astype(np.int)
                self.clean_prob = np.concatenate((self.clean_prob, clean_prob_new[sort_indices]))

                new_num = len(sort_indices)

        self.static_clean(old_len)

        return self.clean_index, new_num


    def static_clean(self, old_len):
        gt = 0
        clean_label_true = []
        if len(self.clean_index) != 0:
            clean_label = self.label[self.clean_index]
            clean_label_true = self.label_true[self.clean_index]
            gt = np.sum(clean_label == clean_label_true)
        print('expend meta form %d to %d, and expend_true is : (%d/%d) %.2f\n' % (old_len, len(self.clean_index), gt,len(self.clean_index), gt/(len(self.clean_index)+0.1)))
        print(f'expend meta labels distribution: {Counter(clean_label_true)}')



    # 只负责输出本次预测的所有干净结果
    class Self_Filtering():

        def __init__(self, datasize, noise_label, confidence_k):
            assert confidence_k >= 1
            self.memory_bank = []
            self.datasize = datasize
            self.noise_label = noise_label
            self.k = confidence_k
            self.counter = 0

        def get_clean_index(self, probs):
            clean_index = []
            self.counter += 1
            result = np.zeros(self.datasize)

            probs_soft = softmax(probs, axis=1)
            max_probs, pred = probs_soft.max(1), probs_soft.argmax(1)
            eq_y_yhat = pred == self.noise_label
            result[eq_y_yhat] = max_probs[eq_y_yhat]

            self.memory_bank.append(result)
            if self.counter < self.k :
                return clean_index, None
            else:
                fluctuation_ = np.ones(self.datasize)
                condition = (result == 0) | ((result != 0) & (self.memory_bank[-2] > result))
                luctuation_index = np.where(condition)
                fluctuation_[luctuation_index] = 0

                confidence_smooth = np.array(self.memory_bank).sum(axis=0)
                self.memory_bank.pop(0)
                prob = (confidence_smooth + fluctuation_) / (self.k + 1)  # adding confidence make fluctuation more smooth
                # 筛选出大于 0.5 的元素的序号
                indices = np.where(prob > 0.5)[0]
                # 根据元素值从大到小对序号进行排序
                clean_index = indices[np.argsort(-prob[indices])]
                return clean_index, prob[clean_index]










