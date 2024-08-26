from evaluate.affiliation import convert_vector_to_events
from evaluate.affiliation import pr_from_events

def getAffiliationMetrics(label, pred):
    events_pred = convert_vector_to_events(pred)
    events_label = convert_vector_to_events(label)
    Trange = (0, len(pred))

    result = pr_from_events(events_pred, events_label, Trange)
    P = result['precision']
    R = result['recall']
    if P + R > 0.0:
        F = 2 * P * R / (P + R)
    else:
        F = 0.0

    return P, R, F


def evaluate(test_label, test_pred, thred):
    res = {
        "threshold": thred
    }
    # affiliation
    precision, recall, f1_score = getAffiliationMetrics(test_label.copy(), test_pred.copy())
    res['P_af'] = precision
    res['R_af'] = recall
    res['F1_af'] = f1_score

    return res