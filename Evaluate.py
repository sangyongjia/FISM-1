import math
import heapq
import numpy as np

def _init_test_data(model, sess, testRating, testNegative, trainList):
    dict_list = []
    for idx in range(len(testRating)):
        rating = testRating[idx]
        items = testNegative[idx]

        user_hist = trainList[rating[0]-1]
        num_idx = len(user_hist)
        gtItem = rating[1]
        items.append(gtItem) #items contains a positive instance and 99 negative instance
        num_idx = np.full(len(items), num_idx,dtype=np.int32)[:,None]
        user_input = []
        for i in range(len(items)):
            user_input.append(user_hist)
        user_input = np.array(user_input)
        item_input = np.array(items)[:,None]
        feed_dict = {model.user_input:user_input, model.item_input:item_input, model.num_idx:num_idx}
        dict_list.append(feed_dict)
    print("<finished>load the evaluate model.")
    return dict_list

def evaluate(model, sess, testRating, testNegative, dict_list):
    hits, ndcgs, losses = [],[],[]

    for idx in range(len(testRating)):
        (hr,ndcg,loss) = _evaluate_one_rating(idx,testRating,testNegative,dict_list,sess,model)
        hits.append(hr)
        ndcgs.append(ndcg)
        losses.append(loss)

    return (hits,ndcgs,losses)

def _evaluate_one_rating(idx,testRating,testNegative,dict_list,sess,model):
    item_scores = {}
    rating = testRating[idx]
    items = testNegative[idx]
    gtItem = rating[1]
    labels = np.zeros(len(items))[:,None]
    labels[-1] = 1
    feed_dict = dict_list[idx]
    feed_dict[model.labels] = labels
    predictions, loss = sess.run([model.output,model.loss],feed_dict=feed_dict)
    for i in range(len(items)):
        item = items[i]
        item_scores[item] = predictions[i]

    rank_list = heapq.nlargest(10,item_scores,key=item_scores.get)
    hr = _get_hit_ratio(rank_list,gtItem)
    ndcg = _get_ndcg(rank_list,gtItem)
    return (hr,ndcg,loss)

def _get_hit_ratio(rank_list, gtItem):
    for i in range(len(rank_list)):
        item = rank_list[i]
        if item == gtItem:
            return 1
    return 0

def _get_ndcg(rank_list, gtItem):
    for i in range(len(rank_list)):
        item = rank_list[i]
        if item == gtItem:
            return math.log(2)/math.log(i+2)
    return 0
