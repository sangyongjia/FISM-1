import numpy as np
from Dataset import Dataset

class Data(object):
    def __init__(self, dataset, num_negative):
        self._Dataset = dataset
        self._num_negative = num_negative
        self._user_input,self._item_input, self._labels,self._batch_length = self._get_train_data_by_user()
        self._num_batch = len(self._batch_length)
        self._batches = self._precoss(self.get_train_batch_by_user)

    def batch_gen(self, i):
        return [(self._batches[r][i]) for r in range(4)]

    def _precoss(self, get_train_batch_by_user):
        user_input_list, item_input_list, num_idx_list,labels_list = [],[],[],[]
        for i in range(self._num_batch):
            ui,ii,ni,l = self.get_train_batch_by_user(i)
            user_input_list.append(ui)
            item_input_list.append(ii)
            num_idx_list.append(ni)
            labels_list.append(l)
        return (user_input_list,item_input_list,num_idx_list,labels_list)

    def _get_train_data_by_user(self):
        #get user_input and item_input that have add negative items
        user_input, item_input, labels, batch_length = [],[],[],[]
        train = self._Dataset.trainMatrix
        train_list = self._Dataset.trainList

        self._num_items,self._num_users = self._Dataset.num_item,self._Dataset.num_user
        for u in range(self._num_users):
            if u == 0:
                batch_length.append((1+self._num_negative)*len(train_list[u]))
            else:
                batch_length.append((1+self._num_negative)*len(train_list[u])+batch_length[u-1])

            for i in train_list[u]:
                #positive instance
                user_input.append(u+1)
                item_input.append(i)
                labels.append(1)
                #negative instance
                for t in range(self._num_negative):
                    j = np.random.randint(self._num_items)
                    while j in train_list[u]:
                        j = np.random.randint(self._num_items)
                    user_input.append(u+1)
                    item_input.append(j)
                    labels.append(0)
        return user_input, item_input, labels, batch_length


    def get_train_batch_by_user(self,i):
        user_list,item_list,num_list,label_list = [],[],[],[]
        trainList = self._Dataset.trainList
        if i == 0:
            begin = 0
        else:
            begin = self._batch_length[i-1]
        batch_index = list(range(begin,self._batch_length[i]))#batch_index 相当于 [user_id-1]
        np.random.shuffle(batch_index)
        for idx in batch_index:
            user = self._user_input[idx]
            item = self._item_input[idx]
            nonzero_row = []
            nonzero_row += trainList[user-1]
            num_list.append(self._remove_item(0, nonzero_row, item))
            user_list.append(nonzero_row)
            item_list.append(item)
            label_list.append(self._labels[idx])
        user_input = np.array(self._add_mask(self._num_items, user_list, 0))
        num_idx = np.array(num_list)
        item_input = np.array(item_list)
        labels = np.array(label_list)
        return (user_input,item_input,num_idx,labels)

    def _remove_item(self, feature_mask, items, item):
        for i in range(len(items)):
            if items[i] == item:
                items[i] = 0
                break
        return len(items) - 1

    def _add_mask(self, feature_mask, user_list, num_max):
        for i in range(len(user_list)):
            user_list[i] = user_list[i] + [feature_mask]*(num_max+1 - len(user_list))
        return user_list


if __name__ == '__main__':
    dataset = Dataset('Data/100k/ua')
    batch_gen = Data(dataset, 4)
    print(batch_gen.batch_gen(1))
