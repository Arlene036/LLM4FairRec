import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score, roc_auc_score

# 读取训练集和测试集
train = pd.read_csv("/Users/lavander/cpan/h411711/dataset/movielens_1M/train_set.csv")
test = pd.read_csv("/Users/lavander/cpan/h411711/dataset/movielens_1M/test_set.csv")

# 构建用户-物品交互矩阵
def create_interaction_matrix(df, user_col, item_col, rating_col):
    user_ids = df[user_col].astype('category').cat.codes
    item_ids = df[item_col].astype('category').cat.codes
    interaction_matrix = csr_matrix((df[rating_col], (user_ids, item_ids)))
    user_mapping = dict(enumerate(df[user_col].astype('category').cat.categories))
    item_mapping = dict(enumerate(df[item_col].astype('category').cat.categories))
    return interaction_matrix, user_mapping, item_mapping

train_matrix, user_mapping, item_mapping = create_interaction_matrix(train, "userId", "movieId", "rating")

# 计算物品相似性矩阵
def compute_item_similarity(train_matrix):
    item_similarity = cosine_similarity(train_matrix.T)  # 转置矩阵以计算物品间的相似性
    return item_similarity

item_similarity = compute_item_similarity(train_matrix)

# 基于物品协同过滤的推荐
def recommend_item_based(user_id, train_matrix, item_similarity, user_mapping, item_mapping, top_n):
    user_idx = {v: k for k, v in user_mapping.items()}[user_id]  # 用户ID转为矩阵索引
    user_interactions = train_matrix[user_idx, :].toarray().flatten()
    scores = user_interactions.dot(item_similarity)  # 计算推荐得分
    scores[user_interactions.nonzero()] = 0  # 排除用户已交互过的物品
    top_items_idx = np.argsort(scores)[::-1][:top_n]  # 得分从高到低排序
    recommended_items = [item_mapping[idx] for idx in top_items_idx]  # 映射回原物品ID
    return recommended_items, scores

# 评估指标
def hit_ratio(predicted, actual):
    return int(any(item in actual for item in predicted))

def precision_at_k(predicted, actual, k):
    relevant = set(predicted[:k]) & set(actual)
    return len(relevant) / k

def recall_at_k(predicted, actual, k):
    relevant = set(predicted[:k]) & set(actual)
    return len(relevant) / len(actual) if actual else 0

def ndcg_at_k(predicted, actual, k):
    relevance = [1 if item in actual else 0 for item in predicted[:k]]
    return ndcg_score([relevance], [sorted(relevance, reverse=True)])

def calculate_auc(test_df, predictions, user_col="userId", item_col="movieId", label_col="rating"):
    user_auc = []
    for user in test_df[user_col].unique():
        actual = test_df[test_df[user_col] == user][label_col]
        pred = predictions[test_df[test_df[user_col] == user].index]
        if len(set(actual)) > 1:  # 需要正负样本
            user_auc.append(roc_auc_score(actual, pred))
    return np.mean(user_auc)

def calculate_accuracy(test_df, predictions, threshold=3.5, user_col="userId", label_col="rating"):
    correct = 0
    total = 0
    for user in test_df[user_col].unique():
        actual = (test_df[test_df[user_col] == user][label_col] >= threshold).astype(int)
        pred = (predictions[test_df[test_df[user_col] == user].index] >= threshold).astype(int)
        correct += (actual == pred).sum()
        total += len(actual)
    return correct / total if total > 0 else 0

# 总评估函数
def evaluate_global_metrics(metrics_results, ks):
    """
    汇总所有用户的指标，计算每个K值下的平均 HitRatio、Precision、Recall、NDCG
    :param metrics_results: 包含每个用户的评估结果
    :param ks: 一个包含 K 值的列表，例如 [3, 5, 10]
    :return: 汇总的评估结果
    """
    global_metrics = {}
    for k in ks:
        hit_ratios = [result[f"HitRatio@{k}"] for result in metrics_results]
        precisions = [result[f"Precision@{k}"] for result in metrics_results]
        recalls = [result[f"Recall@{k}"] for result in metrics_results]
        ndcgs = [result[f"NDCG@{k}"] for result in metrics_results]

        global_metrics[f"HitRatio@{k}"] = np.mean(hit_ratios)
        global_metrics[f"Precision@{k}"] = np.mean(precisions)
        global_metrics[f"Recall@{k}"] = np.mean(recalls)
        global_metrics[f"NDCG@{k}"] = np.mean(ndcgs)
    
    return global_metrics

# 评估用户和全局指标
ks = [3, 5, 10]
metrics_results = []
user_recommendations = []
predictions = np.zeros(test.shape[0])  # 存储所有用户的预测得分
test = test[test.groupby("userId")["movieId"].transform("count") >= 10]

for idx, user in enumerate(test["userId"].unique()):
    # 获取测试集中的真实物品
    user_test_items = test[test["userId"] == user]["movieId"].tolist()
    
    # 基于协同过滤推荐Top-10物品
    recommended_items, user_scores = recommend_item_based(user, train_matrix, item_similarity, user_mapping, item_mapping, 10)
    predictions[idx] = user_scores.mean()  # 存储每用户的平均预测分数

    # 评估指标
    user_metrics = {"userId": user}
    for k in ks:
        user_metrics[f"HitRatio@{k}"] = hit_ratio(recommended_items[:k], user_test_items)
        user_metrics[f"Precision@{k}"] = precision_at_k(recommended_items, user_test_items, k)
        user_metrics[f"Recall@{k}"] = recall_at_k(recommended_items, user_test_items, k)
        user_metrics[f"NDCG@{k}"] = ndcg_at_k(recommended_items, user_test_items, k)
    
    metrics_results.append(user_metrics)
    user_recommendations.append({
        "user_id": user,
        "top10_recommendations": recommended_items,
        "groundtruth": user_test_items
    })

# 汇总评估结果
global_metrics = evaluate_global_metrics(metrics_results, ks)
global_metrics["AUC"] = calculate_auc(test, predictions)
global_metrics["Accuracy"] = calculate_accuracy(test, predictions)

# 打印总评估结果
print("总评估结果：")
for metric, value in global_metrics.items():
    print(f"{metric}: {value:.4f}")

# 保存总评估结果为CSV
global_metrics_df = pd.DataFrame([global_metrics])
global_metrics_df.to_csv("cf/item_cf_global_metrics.csv", index=False)

# 保存推荐结果
recommendations_df = pd.DataFrame(user_recommendations)
recommendations_df.to_csv("cf/item_cf_recommendations.csv", index=False)

print("基于物品协同过滤的总评估和推荐结果已保存。")
