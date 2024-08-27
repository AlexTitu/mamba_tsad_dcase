import torch
import torch.nn as nn
import logging

from baseline_evaluate import evaluate
from evaluate.spot import SPOT


def detect(clf, model_path, test_dataloader, device="cpu",
           init_dataloader=None, pot_params=(0.01, 0.98)):
    # detect
    if isinstance(clf, nn.Module):
        # load model
        model_name = model_path.split("/")[-1]
        best_model = torch.load(model_path)
        clf.load_state_dict(best_model)
        if device != "cpu":
            clf.cuda(0)
    else:
        model_name = model_path
    logging.info("Testing model {} ...".format(model_name))
    # obtain scores
    scores, labels, y_trues, y_hats = clf.anomaly_detection(test_dataloader, device)
    results = {"model": model_name, "scores": scores, "y_trues": y_trues, "y_hats": y_hats, "labels": labels}

    assert init_dataloader is not None
    init_scores, _, _, _ = clf.anomaly_detection(init_dataloader, device)
    q, level = pot_params
    spot = SPOT(init_scores, q=q)
    spot.initialize(level=level)
    th_pot = spot.extreme_quantile
    results.update({"init_scores": init_scores})
    results.update({"th_pot": th_pot})

    pred = (scores > th_pot).astype(int)
    res = evaluate(labels, pred, th_pot)
    results.update(res)

    return results, labels
