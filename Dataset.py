'''
Created on Oct 11, 2018
processing dataset

@author lucas(lucas_hfut@163.com)
'''
import scipy.sparse as sp
import numpy as np
import os

class Dataset(object):
    '''
    load the data
        trainMatrix: load rating data as sparse matrix
        trainList: load rating data as list to speed up user's feature retrieval[[item1,item2..]..]
        testRating: load [[user_id,item_id]..]
        testNegatives: sample the items not rated by user
    '''
    def __init__(self,path):
        self.trainMatrix = self.load_training_file_as_matrix(path+'.base')

        self.trainList = self.load_training_file_as_list(path+'.base')
        self.testRatings = self.load_test_file_as_list(path+'.test')
        self.num_user, self.num_item = self.trainMatrix.shape[0]-1,self.trainMatrix.shape[1]-1
        self.testNegatives= self.generate_negative_file_from_test_file(path+'.test',path+'.negative')
        assert len(self.testRatings)==len(self.testNegatives)


    def load_training_file_as_matrix(self, filename):
        num_user, num_item = 0,0
        with open(filename,'r') as file:
            for line in file:
                if line == None or line == '':
                    continue
                arr = line.split('\t')
                u,i = int(arr[0]),int(arr[1])
                num_user = max(num_user,u)
                num_item = max(num_item,i)
        #construct sparse matrix
        mat = sp.dok_matrix((num_user+1,num_item+1),dtype=np.float32)
        with open(filename,'r') as file:
            for line in file:
                if line == None or line == '':
                    continue
                arr = line.split('\t')
                user_,item_,rating = int(arr[0]),int(arr[1]),int(arr[2])
                if rating > 0 :
                    mat[user_,item_] = 1.0
        print('<finished>load the trainMatrix')
        return mat

    def load_training_file_as_list(self,filename):
        #get the itemlist of all user
        with open(filename,'r') as file:
            u_begin = 1
            items = []
            item_list = []
            for line in file:
                if line == None or line == '':
                    continue
                arr = line.split('\t')
                user,item = int(arr[0]),int(arr[1])
                if user == u_begin:
                    item_list.append(item)
                else:
                    items.append(item_list)
                    item_list = []
                    item_list.append(item)
                    u_begin = user
            items.append(item_list)
        print('<finished>load the trainList')
        # print(items[0])
        return items

    def load_test_file_as_list(self,filename):
        #get the [user,item] from test file
        rating_list = []
        with open(filename,'r') as file:
            for line in file:
                if line == None or line == '':
                    continue
                arr = line.split('\t')
                user_, item_ = int(arr[0]),int(arr[1])
                rating_list.append([user_, item_])
        print('<finished>load the testRating')
        return rating_list


    def generate_negative_file_from_test_file(self,filename,neg_filename):
        #generate the [[N-item1,N-item2..]..] from test file
        N_list = []
        N_items = []
        if os.path.exists(neg_filename):
            with open(neg_filename) as file:
                for line in file:
                    if line == None or line == '':
                        continue
                    items = line.strip().split('\t')
                    for item in items:
                        N_items.append(int(item))
                    N_list.append(N_items)
                    N_items = []
            print('<finished>load the negative list(have negative file)')
            return N_list



        with open(filename,'r') as file:
            for line in file:
                if line == None or line == '':
                    continue
                arr = line.split('\t')
                user_, item_ = int(arr[0]),int(arr[1])
                count = 0
                item_list = self.trainList[user_ - 1]
                item_index = np.arange(1,self.num_item+1)
                # np.random.shuffle(item_index)
                for item in item_index:
                    if item not in item_list:
                        count += 1
                        N_items.append(item)
                    if count == 99:
                        N_list.append(N_items)
                        N_items = []
                        break
        with open(neg_filename,'w') as f:
            for list in N_list:
                for item in list:
                    f.write(str(item)+'\t')
                f.write('\n')
        print('<finished>generate the negative list(no negative file)')
        return N_list


if __name__ == '__main__':
    dataset = Dataset('Data/100k/ua')
    print(len(dataset.trainList))
