import numpy as np
import torch
import os
from model import RippleNet


def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    model = RippleNet(args, n_entity, n_relation)
    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )

    log_file_path = "./training_log.txt"
    save_path = "./checkpoints/"
    results_path = "./results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Open log file
    log_file = open(log_file_path, "w")
    best_eval_auc = 0
    best_model_path = None

    for step in range(args.n_epoch):
        print(f"Epoch {step}: Model is on device: {next(model.parameters()).device}")
        items, labels, memories_h, memories_r, memories_t = get_feed_dict(
            args, model, train_data, ripple_set, 0, args.batch_size
        )
        print(f"Sample data batch is on device: {items.device}")
        
        # training
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += args.batch_size
            if show_loss:
                # print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss.item()))
                log_file.write('%.1f%% %.4f\n' % (start / train_data.shape[0] * 100, loss.item()))

        # evaluation
        train_auc, train_acc = evaluation(args, model, train_data, ripple_set, args.batch_size)
        eval_auc, eval_acc = evaluation(args, model, eval_data, ripple_set, args.batch_size)
        test_auc, test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)

        # print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
        #         % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))
        log_message = (
            f"Epoch {step}: Train AUC: {train_auc:.4f}, Train ACC: {train_acc:.4f}, "
            f"Eval AUC: {eval_auc:.4f}, Eval ACC: {eval_acc:.4f}, "
            f"Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.4f}\n"
        )
        print(log_message)
        log_file.write(log_message)

        if eval_auc > best_eval_auc:
            best_eval_auc = eval_auc
            best_model_path = os.path.join(save_path, f"best_model_epoch_{step}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {step} with Eval AUC: {eval_auc:.4f}")
            log_file.write(f"Best model saved at epoch {step} with Eval AUC: {eval_auc:.4f}\n")

    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path} for testing...")
        log_file.write(f"Loaded best model from {best_model_path} for testing...\n")

        final_test_auc, final_test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)
        print(f"Final Test AUC: {final_test_auc:.4f}, Final Test ACC: {final_test_acc:.4f}")
        log_file.write(f"Final Test AUC: {final_test_auc:.4f}, Final Test ACC: {final_test_acc:.4f}\n")

        # Save test predictions
        save_predictions(args, model, test_data, ripple_set, os.path.join(results_path, "test_predictions.txt"))
        save_top_k_recommendations(args, model, test_data, ripple_set, os.path.join(results_path, "top_k_recommendations.txt"))

    log_file.close()

def get_feed_dict(args, model, data, ripple_set, start, end):
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])
    memories_h, memories_r, memories_t = [], [], []
    for i in range(args.n_hop):
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
    if args.use_cuda:
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return items, labels, memories_h, memories_r,memories_t


def evaluation(args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    model.eval()
    while start < data.shape[0]:
        auc, acc = model.evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    model.train()
    return float(np.mean(auc_list)), float(np.mean(acc_list))

def save_predictions(args, model, data, ripple_set, output_file):
    """Save predictions (probabilities) for the test set."""
    start = 0
    predictions = []
    model.eval()
    while start < data.shape[0]:
        items, labels, memories_h, memories_r, memories_t = get_feed_dict(args, model, data, ripple_set, start, start + args.batch_size)
        scores = model(*[items, labels, memories_h, memories_r, memories_t])["scores"].detach().cpu().numpy()
        predictions.extend(scores.tolist())
        start += args.batch_size
    model.train()
    with open(output_file, "w") as f:
        for score in predictions:
            f.write(f"{score}\n")
    print(f"Predictions saved to {output_file}")


def save_top_k_recommendations(args, model, data, ripple_set, output_file, k=10):
    """Save top-k recommendations for each user."""
    user_recommendations = {}
    model.eval()
    start = 0
    while start < data.shape[0]:
        items, labels, memories_h, memories_r, memories_t = get_feed_dict(args, model, data, ripple_set, start, start + args.batch_size)
        scores = model(*[items, labels, memories_h, memories_r, memories_t])["scores"].detach().cpu().numpy()
        users = data[start:start + args.batch_size, 0]
        for user, score, item in zip(users, scores, data[start:start + args.batch_size, 1]):
            if user not in user_recommendations:
                user_recommendations[user] = []
            user_recommendations[user].append((item, score))
        start += args.batch_size
    model.train()

    # Sort recommendations and save top-k
    with open(output_file, "w") as f:
        for user, items_scores in user_recommendations.items():
            top_k = sorted(items_scores, key=lambda x: -x[1])[:k]
            f.write(f"User {user}: {top_k}\n")
    print(f"Top-K recommendations saved to {output_file}")