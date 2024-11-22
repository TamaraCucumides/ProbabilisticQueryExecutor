import os
import pickle
import torch
import pprint

import utils
import solve_queries

current_dir = "/home/tacucumides/storage/nbfnet-experiments/KnowledgeGraphCompletionV2/FB15k237-2/NBFNet/scripts/"
query_dir = "/home/tacucumides/storage/nbfnet-experiments/KnowledgeGraphCompletionV2/FB15k237-2/NBFNet/queries/"


def load_queries(path=query_dir):
    with open(path + '100-test-queries.pkl', 'rb') as f:
        queries = pickle.load(f)

    with open(path + '100-test-easy-answers.pkl', 'rb') as f:
        easy = pickle.load(f)

    with open(path + '100-test-hard-answers.pkl', 'rb') as f:
        hard = pickle.load(f)

    count = 0

    for key in queries.keys():
        for query in queries[key]:
            if queries[key] and (len(easy[query]) == 0 or len(hard[query]) == 0):
                print("Query con problemas")
                queries[key].remove(query)
                count +=1
    print(count, "deleted queries")

    for k,v in queries.items():
        print("Query type", k, "-- N Queries: ", len(v))

    
    return queries, easy, hard

def evaluate(queries, preds, easies, hards, metrics, thresh):
    mask_size = preds[queries[0]].shape[0]
    metrics_dict = {}
    for metric in metrics:
        if metric == "spearmanr":
            #0.5 is not a very good thresh for all queries
            pass

        elif metric == "avg":
            pass

        elif metric in ["Precision", "Recall"]:
            for t in thresh:
                metric_values = []
                for query in queries:
                    if len(easies[query])== 0 or len(hards[query]) == 0:
                        print("Query sin easies o hards")
                        continue
                    pred = preds[query]
                    easy = utils.mask_answers(easies[query], mask_size)
                    hard = utils.mask_answers(hards[query], mask_size)
                    if metric == "Precision":
                        predicted_ans = pred > t
                        number_predicted = predicted_ans.sum(dim=-1)
                        if number_predicted == 0:
                            metric_values.append(0)
                        else:
                            true_positive = (predicted_ans * (torch.logical_or(easy, hard))).sum(dim=-1)
                            query_score = (true_positive / number_predicted).float()
                            metric_values.append(query_score)
                    elif metric == "Recall":
                        # This is recall over hard answers
                        predicted_ans = pred > t
                        number_predicted = predicted_ans.sum(dim=-1)
                        hard_answers = hard.sum(dim=-1)
                        true_positive_hard = (predicted_ans * hard).sum(dim=-1)
                        query_score = (true_positive_hard / hard_answers).float()
                        metric_values.append(query_score)

                        
                metrics_dict[metric+f"@{t}"]= sum(metric_values)/len(metric_values)
        else:
            metric_values = [] 
            for query in queries:
                if len(easies[query])== 0 or len(hards[query]) == 0:
                    print("Query con problemas en evaluaciòn")
                    continue

                pred = preds[query]
                easy = utils.mask_answers(easies[query], mask_size)
                hard = utils.mask_answers(hards[query], mask_size)
                #mask out the easy answers from pred
                pred -= easy * pred
                # remove irrelevant entities
                pred[14505:14541] = 0

                if metric == "mrr":
                    masked_preds = pred[hard]
                    max_hard = torch.max(masked_preds)
                    ranking = torch.sum((max_hard <= pred)) + 1
                    metric_values.append(1/ranking)

                    # _, indices = torch.sort(pred, descending=True)
                    # ranking = torch.argsort(indices) + 1
                    # masked_ranking = ranking[hard]
                    # min_rank = torch.min(masked_ranking)
                    # metric_values.append(1/min_rank)
                elif metric.startswith("hits@"):
                    k = int(metric[5:])
                    _, ranking = torch.sort(pred, descending=True)
                    top_k_mask = torch.zeros_like(pred, dtype=torch.bool)
                    top_k_mask[ranking[:k]] = True

                    intersection_count = torch.sum(top_k_mask & hard)
                    percentage = intersection_count.item() / k

                    metric_values.append(percentage)

                elif metric == "mape":
                    predicted_ans = pred > 0.5
                    num_pred = predicted_ans.sum(dim=-1)
                    num_easy = len(easies[query])
                    num_hard = len(hards[query])

                    mape = (num_pred - num_easy - num_hard).abs() / (num_easy + num_hard).float()

                    metric_values.append(mape)
                else:
                    print("Unknown metric", metric)

            metrics_dict[metric] = sum(metric_values)/len(metric_values)

    return metrics_dict


if __name__ == "__main__":
    queries, easy, hard = load_queries()

    struct2type = utils.struct2type

    metrics = ["mrr", "hits@1", "hits@3", "hits@10",
    "Precision", "Recall"]

    thresh = [0.1, 0.3, 0.5, 0.7, 0.9]

    for struct, tipo in struct2type.items():
        try:
            print("### Query type", tipo, "###")
            qs = list(queries[struct])[:10]
            # Evaluate the queries
            preds = solve_queries.solve_query(tipo, qs)
            metrics_dict = evaluate(qs, preds, easy, hard, metrics, thresh)
            for metric, value in metrics_dict.items():
                print(metric, value)
            print("#########################")
        except KeyError:
            pass