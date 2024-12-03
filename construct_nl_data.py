# input are two lists: (1) groudtruth of mappedmovie IDs (not original IDs)
#  (2) top 10 recommended items (movie ID before mapping back to original ID space)
import pickle
import pandas as pd
import re
import argparse
def convert_items_to_original_ids(processed_user_id, groundtruth_item_ids, top10_item_ids):
    """
    Args:
        processed_user_id: int
        processed_item_ids: list
    
    Returns:
        original_user_id: int
        original_item_ids: list
    """
    with open('../data/item_map.pkl', 'rb') as f:
        item_map = pickle.load(f)
    with open('../data/user_map.pkl', 'rb') as f:
        user_map = pickle.load(f)
    item_new2old = {v: k for k, v in item_map.items()}
    user_new2old = {v: k for k, v in user_map.items()}
    original_gt_item_ids = [item_new2old[pid] for pid in groundtruth_item_ids]
    original_top10_item_ids = [item_new2old[pid] for pid in top10_item_ids]
    original_user_id = user_new2old[processed_user_id]
    return original_user_id, original_gt_item_ids, original_top10_item_ids


def generate_sequential_dataset(input_data,
                                train_set_path='../dataset/movielens_1M/train_set.csv',
                                seq_length=10):

    user_id = input_data['user_id'].tolist()
    groundtruth = input_data['groundtruth'].tolist()
    top10_recommendations = input_data['top10_recommendations'].tolist()
    train_set = pd.read_csv(train_set_path)
    original_user_id, original_gt_item_ids, original_top10_item_ids = convert_items_to_original_ids(user_id, groundtruth, top10_recommendations)

    dataset = []
    for uid, gt_list, top10_list in zip(original_user_id, original_gt_item_ids, original_top10_item_ids):
        user_history = train_set[train_set['userId'] == uid]['movieId'].tolist()
        # sort by timestamp
        user_history.sort(key=lambda x: train_set[train_set['movieId'] == x]['timestamp'].values[0])
        if len(user_history) > seq_length:
            user_history = user_history[-seq_length:]
        
        entry = {
            'userId': uid,
            'history': user_history,
            'candidates': top10_list,
            'groundtruth': gt_list
        }
        dataset.append(entry)
    return pd.DataFrame(dataset)

def get_movie_description(row):
    # Ensure genres is properly processed
    genres = row['genres']
    if isinstance(genres, str):  # If genres are stored as a string, evaluate it to a list
        genres = eval(genres)
    if not isinstance(genres, list):  # Ensure genres is a list
        genres = [str(genres)]

    return f"Movie {row['movieId']}: {row['title']} (Genres: {', '.join(genres)}; " \
           f"Language: {row['original_language']}; Overview: {row['short_overview']})"

def process_sequence(row):
    history = eval(row['history'])
    history_movies = filtered_movies[filtered_movies['movieId'].isin(history)].apply(get_movie_description, axis=1).tolist()
    
    candidates = eval(row['candidates'])
    candidate_movies = filtered_movies[filtered_movies['movieId'].isin(candidates)].apply(get_movie_description, axis=1).tolist()
    
    next_items = eval(row['groundtruth'])
    ground_truth = filtered_movies[filtered_movies['movieId'].isin(next_items)].apply(get_movie_description, axis=1).tolist()
    
    return pd.Series({
        'user_id': row['userId'],
        'history': ','.join(history_movies),
        'candidates': ','.join(candidate_movies), 
        'ground_truth': ','.join(ground_truth)
    })

def build_natural_language_dataset(seq_dataset, 
                                   big_table="dataset/movielens_1M/used_movies.csv",
                                   n = 100):
    """
    Args:
        seq_dataset: pd.DataFrame, sequential dataset
        big_table: str, path to the big table
        n: int, number of samples to process (how large the dataset is for inference!!!)
    """
    global filtered_movies
    filtered_movies = pd.read_csv(big_table)
    if n is not None:
        natural_language_dataset = seq_dataset[:n].apply(process_sequence, axis=1)
    else:
        natural_language_dataset = seq_dataset.apply(process_sequence, axis=1)
    return natural_language_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../dataset/movielens_1M/test_set.csv", 
                        help='3 column dataset for inference: user_id, groundtruth, top10_recommendations')
    parser.add_argument("--output_path", type=str, default="natural_language_dataset.csv", 
                        help='natural language dataset for further inference')
    parser.add_argument("--n", type=int, default=None, 
                        help='number of samples to process (how large the dataset is for inference!!!)')
    args = parser.parse_args()

    # read the input file
    input_data = pd.read_csv(args.input_path)
    seq_dataset = generate_sequential_dataset(input_data, n=args.n)
    natural_language_dataset = build_natural_language_dataset(seq_dataset)
    natural_language_dataset.to_csv(args.output_path, index=False)
