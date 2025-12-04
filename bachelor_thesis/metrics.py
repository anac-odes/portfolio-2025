# This file containts a function for computing metrics I will use to evaluate quickscore

from sklearn.metrics import log_loss
import numpy as np

# New metrics, with 0 instead of np.nan
def compute_metrics(actual, posterior, threshold=0.5):
    """
    Compute classification metrics and log loss for quickscore predictions
    """
    predictions = (posterior >= threshold).astype(int)

    TP = np.sum((predictions == 1) & (actual == 1))
    FN = np.sum((predictions == 0) & (actual == 1))
    FP = np.sum((predictions == 1) & (actual == 0))
    TN = np.sum((predictions == 0) & (actual == 0))

    # WW precision: what if it didn't found any of the existing positives? => bad job => 0
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    elif (TP + FN )> 0:
        precision = 0
    else:
        precision = np.nan
    
    # recall 
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    
    # WW in f1, define f1=0 if both precision and recall are zero (this is a natural extension of your definition)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    try:
        logloss = log_loss(actual, posterior, labels=[0, 1])
    except ValueError:
        logloss = np.nan
    
    return {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "precision": precision,
        "recall": recall,
         "f1_score": f1_score,
        "log_loss": logloss
    }


# Old metrics

# def compute_metrics(actual, posterior, threshold=0.5):
#     """
#     Compute classification metrics and log loss for quickscore predictions
#     """
#     predictions = (posterior >= threshold).astype(int)

#     TP = np.sum((predictions == 1) & (actual == 1))
#     FN = np.sum((predictions == 0) & (actual == 1))
#     FP = np.sum((predictions == 1) & (actual == 0))
#     TN = np.sum((predictions == 0) & (actual == 0))

#     precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
#     recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

#     try:
#         logloss = log_loss(actual, posterior, labels=[0, 1])
#     except ValueError:
#         logloss = np.nan
    
#     return {
#         "TP": TP, "FN": FN, "FP": FP, "TN": TN,
#         "precision": precision,
#         "recall": recall,
#          "f1_score": f1_score,
#         "log_loss": logloss
#     }