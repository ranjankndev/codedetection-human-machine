from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def print_classification_report(y_true, y_pred):
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, digits=4))
