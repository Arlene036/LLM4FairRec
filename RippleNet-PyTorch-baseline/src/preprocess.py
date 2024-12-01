import argparse
import numpy as np
import pickle
RATING_FILE_NAME = dict({'movie': 'ratings.dat', 'book': 'BX-Book-Ratings.csv', 'news': 'ratings.txt'})
SEP = dict({'movie': '::', 'book': ';', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'news': 0})


def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))

    # 返回映射字典
    return user_index_old2new, item_index_old2new


def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')

    files = []
    if DATASET == 'movie':
        files.append(open('../data/' + DATASET + '/kg_part1_rehashed.txt', encoding='utf-8'))
        files.append(open('../data/' + DATASET + '/kg_part2_rehashed.txt', encoding='utf-8'))
    else:
        files.append(open('../data/' + DATASET + '/kg_rehashed.txt', encoding='utf-8'))

    for file in files:
        for line in file:
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


def convert_processed_to_original(processed_array, user_index_old2new, item_index_old2new):
    """
    将处理后的user-item-interaction数组转换回原始ID
    
    Args:
        processed_array: numpy array, 形状为 (N, 3)，包含 [processed_user_id, processed_item_id, interaction]
        user_index_old2new: dict, 用户ID的原始到处理后的映射字典
        item_index_old2new: dict, 物品ID的原始到处理后的映射字典
    
    Returns:
        original_array: 包含原始ID的数组
    """
    # 创建反向映射字典
    user_new2old = {v: k for k, v in user_index_old2new.items()}
    
    # 为item创建反向映射
    item_new2old = {}
    for old_id, new_id in item_index_old2new.items():
        item_new2old[new_id] = old_id
    
    # 转换数组
    original_array = processed_array.copy()
    for i in range(len(processed_array)):
        # 转换用户ID
        original_array[i, 0] = user_new2old[processed_array[i, 0]]
        # 转换物品ID
        original_array[i, 1] = item_new2old[processed_array[i, 1]]
    
    return original_array


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.dataset

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()

    user_map, item_map = convert_rating()
    # save user_map and item_map
    with open('../data/user_map.pkl', 'wb') as f:
        pickle.dump(user_map, f)
    with open('../data/item_map.pkl', 'wb') as f:
        pickle.dump(item_map, f)
    print('begin')
    train_data = np.load('../data/train_data.npy', allow_pickle=True)
    test_data = np.load('../data/test_data.npy', allow_pickle=True)
    eval_data = np.load('../data/eval_data.npy', allow_pickle=True)

    train_new = convert_processed_to_original(train_data, user_map, item_map)
    test_new = convert_processed_to_original(test_data, user_map, item_map)
    eval_new = convert_processed_to_original(eval_data, user_map, item_map)
    # 保存转换后的数据
    np.save('../data/train_data_original.npy', train_new)
    np.save('../data/test_data_original.npy', test_new) 
    np.save('../data/eval_data_original.npy', eval_new)
    # convert_kg()

    print('done')
