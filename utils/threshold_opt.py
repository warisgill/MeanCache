from .eutil import evaluateTransformerModel
from tqdm import tqdm
from sentence_transformers import  evaluation
# from bayes_opt import BayesianOptimization


# def findThresholdBinarySearch(model, df, low=0, high=1, eps=1e-3):
#     optimal_threshold = low
#     best_f1 = -1

#     while high - low > eps:
#         mid = (high + low) / 2

#         r_low = evaluateTransformerModel(
#             model,
#             df["question1"].tolist(),
#             df["question2"].tolist(),
#             df["is_duplicate"].tolist(),
#             low,
#         )
#         r_mid = evaluateTransformerModel(
#             model,
#             df["question1"].tolist(),
#             df["question2"].tolist(),
#             df["is_duplicate"].tolist(),
#             mid,
#         )
#         r_high = evaluateTransformerModel(
#             model,
#             df["question1"].tolist(),
#             df["question2"].tolist(),
#             df["is_duplicate"].tolist(),
#             high,
#         )

#         if r_mid["F1"] > best_f1:
#             best_f1 = r_mid["F1"]
#             optimal_threshold = mid

#         if r_low["F1"] > r_high["F1"]:
#             high = mid
#         else:
#             low = mid

#     return optimal_threshold, best_f1






# def thresholdTernarySearch(model, df):
#     low = 0
#     high = 1
#     optimal_threshold = low
#     f1 = -1
#     precision = 0.0001  # Precision of the solution

#     while high - low > precision:
#         mid1 = low + (high - low) / 3
#         mid2 = high - (high - low) / 3

#         f1_mid1 = evaluateTransformerModel(
#             model,
#             df["question1"].tolist(),
#             df["question2"].tolist(),
#             df["is_duplicate"].tolist(),
#             mid1,
#         )["F1"]

#         f1_mid2 = evaluateTransformerModel(
#             model,
#             df["question1"].tolist(),
#             df["question2"].tolist(),
#             df["is_duplicate"].tolist(),
#             mid2,
#         )["F1"]

#         if f1_mid1 > f1_mid2:
#             high = mid2
#             if f1_mid1 > f1:
#                 f1 = f1_mid1
#                 optimal_threshold = mid1
#         else:
#             low = mid1
#             if f1_mid2 > f1:
#                 f1 = f1_mid2
#                 optimal_threshold = mid2

#     return optimal_threshold



# def evaluate_threshold(threshold):
#     r = evaluateTransformerModel(
#         model,
#         df["question1"].tolist(),
#         df["question2"].tolist(),
#         df["is_duplicate"].tolist(),
#         threshold,
#     )
#     return r["F1"]

# # Bounded region of parameter space
# pbounds = {'threshold': (0, 1)}

# optimizer = BayesianOptimization(
#     f=evaluate_threshold,
#     pbounds=pbounds,
#     random_state=1,
# )

# optimizer.maximize(
#     init_points=2,
#     n_iter=10,
# )

# print(optimizer.max)

def findThreshold(model, val_data, hit_miss, region=(50,90)):
    thresholds = []
    results = []
    optimal_threshold = -1
    f1 = -1

    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(*val_data, batch_size=128)
    

    res = binary_acc_evaluator.compute_metrices(model)
    # print('res: ', res['cossim'])
    sbert_threshold = res['cossim']['f1_threshold']

    # for i in tqdm(range(*region)):
    #     threshold = i / 100
    #     # print(f"threshold: {threshold}")
    #     r = evaluateTransformerModel(
    #         model,
    #         df["question1"].tolist(),
    #         df["question2"].tolist(),
    #         df["is_duplicate"].tolist(),
    #         threshold,
    #         hit_miss=hit_miss,
    #     )

    #     if r["F1"] > f1:
    #         f1 = r["F1"]
    #         optimal_threshold = threshold

    #     thresholds.append(threshold)
    #     results.append(r)

    


    return sbert_threshold