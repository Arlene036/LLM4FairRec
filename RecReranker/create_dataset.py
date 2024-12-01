import numpy as np
import pandas as pd
import pickle

def random_candidates(movie_ids, interacted_items, gt_items,
                      cand_samples=100,
                      cand_length=10):
    # 一共sample cand_samples 个候选集
    # 从groupth 和 非gt中分别sample，比例在0.1 - 0.8之间正太分布
    candidates_list = []
    target_items = list(set(movie_ids) - set(gt_items) - set(interacted_items))
    for _ in range(cand_samples):
        gt_ratio = np.random.normal(0.4, 0.15)
        if gt_ratio < 0.1:
            gt_ratio = 0.1
        elif gt_ratio > 0.8:
            gt_ratio = 0.8

        gt_samples = int(cand_length * gt_ratio)
        if gt_samples < 1:
            gt_samples = 1
        non_gt_samples = cand_length - gt_samples

        if len(gt_items) < gt_samples:
            gt_samples = len(gt_items)
        gt_candidates = np.random.choice(gt_items, gt_samples, replace=False)
        
        non_gt_candidates = np.random.choice(target_items, non_gt_samples, replace=False)

        candidates = np.concatenate([gt_candidates, non_gt_candidates])
        # shuffle
        np.random.shuffle(candidates)
        candidates_list.append(candidates)
    return candidates_list


def create_sequential_dataset(train_set, test_set, movie_ids,
                              candidates_generator:str = 'random',
                              seq_length=10, 
                              cand_length=10,
                              cand_samples=100):
    """
    Create sequential recommendation dataset
    
    Parameters:
    - train_set: Training DataFrame (userId, movieId, rating, timestamp)
    - test_set: Testing DataFrame
    - movie_ids: All movie ids
    - candidates_generator: Candidates generator method; chosen from ['random', ''] (TODO)
    - seq_length: Targeted Length of historical interaction sequence
    - his_samples: Number of historical interaction sequences to sample for each user
    - cand_samples: Number of candidate items to sample for each user
    """
    dataset = []
    
    user = train_set.groupby('userId')
    
    for uid, group in user:
        sorted_items = group.sort_values('timestamp')['movieId'].tolist()
        
        if len(sorted_items) < seq_length:
            continue
                        
        test_items = test_set[test_set['userId']==uid]['movieId'].tolist()
        if len(test_items) < 10:
            test_items = test_items
        else:
            test_items = test_items[:10]
        
        if candidates_generator == 'random':
            candidates_list = random_candidates(movie_ids, sorted_items, test_items, cand_samples, cand_length)
        else:
            raise ValueError(f"Invalid candidates generator: {candidates_generator}")


        for candidates in candidates_list:
            his_len = len(sorted_items)
            if his_len > seq_length:
                selected_idx = np.random.choice(his_len, seq_length, replace=False)
                sorted_selected_idx = np.sort(selected_idx)
                seq = [sorted_items[i] for i in sorted_selected_idx]
            else:
                seq = sorted_items
                break
            
            entry = {
                'userId': uid,
                'history': seq,
                'candidates': candidates.tolist(),
                'groundtruth': test_items
            }
            dataset.append(entry)
    
    return pd.DataFrame(dataset) 


if __name__ == '__main__':
    train_set = pd.read_csv('../dataset/movielens_1M/train_set.csv')
    test_set = pd.read_csv('../dataset/movielens_1M/test_set.csv')
    # load movie_ids
    with open('../dataset/movielens_1M/movie_ids.pkl', 'rb') as f:
        movie_ids = pickle.load(f)

    for cand_length_list in [3, 5, 10]:
        dataset = create_sequential_dataset(train_set, test_set, movie_ids, 
                                            candidates_generator='random', 
                                            cand_length=cand_length_list,
                                            cand_samples=30)
        dataset.to_csv(f'../dataset/movielens_1M/sequential_top{cand_length_list}.csv', index=False)
