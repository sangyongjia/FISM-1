import tensorflow as tf
import numpy as np
import logging

from time import time
from Dataset import Dataset
from Batch_gen import Data
import Evaluate as evaluate

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run FISM.")
    parser.add_argument('--path', nargs='?', default='Data/100k/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ua',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed:batch_size: generate batches by batch size')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[1e-7,1e-7]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=1,
                        help='Caculate training loss or not')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    return parser.parse_args()


class FISM:
    def __init__(self, num_items, args):
        self.num_items = num_items
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.alpha = args.alpha
        self.verbose = args.verbose
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.batch_choice = args.batch_choice
        self.train_loss = args.train_loss
        self.build_graph()

    def _create_placeholder(self):
        with tf.name_scope('input_data'):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])
            self.item_input = tf.placeholder(tf.int32, shape=[None,1])
            self.num_idx = tf.placeholder(tf.float32, shape=[None,1])
            self.labels = tf.placeholder(tf.float32, shape=[None,1])

    def _create_inference(self):
        with tf.name_scope('inference'):
            self.c1 = tf.constant(0.0,tf.float32,[1,self.embedding_size],name='c1')
            self.c2 = tf.Variable(tf.truncated_normal(shape=[self.num_items,self.embedding_size],mean=0.0,stddev=0.01),name='c2',dtype=tf.float32)
            self.c3 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),name='c3', dtype=tf.float32)

            self.embedding_P = tf.concat([self.c1,self.c2],0,name='embedding_P')
            self.embedding_Q = tf.concat([self.c1,self.c3],0,name='embedding_Q')
            self.bias = tf.Variable(tf.zeros(self.num_items+1),name='bias')

            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P,self.user_input),1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q,self.item_input),1)
            self.bias_i = tf.nn.embedding_lookup(self.bias,self.item_input)
            self.coeff = tf.pow(self.num_idx, -tf.constant(self.alpha,tf.float32,[1]))
            self.output = tf.sigmoid(self.coeff*tf.expand_dims(tf.reduce_sum(self.embedding_p*self.embedding_q,1),1)+self.bias_i)

            # self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_items + 1, self.embedding_size], mean=0.0, stddev=0.01),name='embedding_P', dtype=tf.float32)
            # self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items + 1, self.embedding_size], mean=0.0, stddev=0.01), name='embedding_Q', dtype=tf.float32)
            # self.bias = tf.Variable(tf.zeros(self.num_items + 1), name='bias')
            # self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            # self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)
            # self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            # self.coeff = tf.pow(self.num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            # self.output = tf.sigmoid(self.coeff * tf.expand_dims(tf.reduce_sum(self.embedding_p * self.embedding_q, 1), 1) + self.bias_i)

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.losses.log_loss(self.labels , self.output) + self.lambda_bilinear*tf.reduce_sum(tf.square(self.embedding_P))+ self.gamma_bilinear*tf.reduce_sum(tf.square(self.embedding_Q) + self.gamma_bilinear*tf.reduce_sum(tf.square(self.bias_i)))

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholder()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        print('<finished> build the tensorflow graph')

def training(model, dataset, epochs, num_negative, filename):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_begin = time()
        _Data = Data(dataset, num_negative)
        batches = _Data._batches
        batch_time = time() - batch_begin

        num_batch = len(batches[2])
        batch_index = list(range(num_batch))

        #initialize the evaluation feed dicts
        testDict = evaluate._init_test_data(model, sess, dataset.testRatings, dataset.testNegatives,
                                                dataset.trainList)

        best_hr, best_ndcg = 0.0, 0.0
        for epoch_count in range(epochs):
            train_begin = time()
            training_batch(batch_index, model, sess, _Data)
            train_time = time() - train_begin

            loss_begin = time()
            train_loss = training_loss(batch_index, model, sess, _Data)
            loss_time = time() - loss_begin

            eval_begin = time()
            (hits, ndcgs, losses) = evaluate.evaluate(model, sess, dataset.testRatings, dataset.testNegatives, testDict)
            hr, ndcg, test_loss = np.array(hits).mean(),np.array(ndcgs).mean(),np.array(losses).mean()
            eval_time = time() - eval_begin

            if hr > best_hr:
                best_hr = hr
                best_ndcg = ndcg

            print("Epoch {} [{:.1f}s + {:.1f}s]: HR = {:.4f}, NDCG = {:.4f}, loss ={:.4f} [{:.4f}s] train_loss ={:.4f} [{:.1f}s]".format(
                epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))
            with open(filename, 'a+') as f:
                f.write("Epoch {} [{:.1f}s + {:.1f}s]: HR = {:.4f}, NDCG = {:.4f}, loss ={:.4f} [{:.4f}s] train_loss ={:.4f} [{:.1f}s]\n".format(
                        epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))
            f.close()
            np.random.shuffle(batch_index)
        return best_hr,best_ndcg


def training_batch(batch_index,model,sess,_Data):
    for index in batch_index:
        user_input, item_input, num_idx, labels = _Data.batch_gen(index)
        feed_dict = {model.user_input:user_input, model.item_input:item_input[:,None], model.num_idx:num_idx[:,None],model.labels:labels[:,None]}
        sess.run(model.optimizer, feed_dict)

def training_loss(batch_index, model, sess, _Data):
    train_loss = 0.0
    for index in batch_index:
        user_input, item_input, num_idx, labels = _Data.batch_gen(index)
        feed_dict = {model.user_input:user_input, model.item_input:item_input[:,None], model.num_idx:num_idx[:,None], model.labels:labels[:,None]}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / len(batch_index)


if __name__ == '__main__':
    args = parse_args()
    regs = eval(args.regs)

    filename = 'result/100k_index_no_shuffle_2.txt'
    with open(filename, 'a+') as f:
        f.write('lr = {}\talpha = {}\tembedding_size = {}\treg = {}\n'.format(args.lr, args.alpha, args.embed_size, args.regs))
    f.close()

    dataset = Dataset(args.path+args.dataset)
    model = FISM(dataset.num_item, args)
    best_hr,best_ndcg = training(model,dataset,args.epochs,args.num_neg,filename)
    print('END. best_HR = {:.4f},best NDCG = {:.4f}'.format(best_hr,best_ndcg))

    with open(filename, 'a+') as f:
        f.write('best HR = {:.4f}, best NDCG = {:.4f}\n'.format(best_hr, best_ndcg))
        f.write('\n\n')
    f.close()