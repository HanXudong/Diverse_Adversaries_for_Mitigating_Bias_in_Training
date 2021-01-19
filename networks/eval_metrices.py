from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle


def group_evaluation(preds, labels, p_labels, silence=True):

    preds = np.array(preds)
    labels = np.array(labels)
    p_labels = np.array(p_labels)

    p_set = set(p_labels)
    assert len(p_set)==2, "Assuming binary private labels"

    g1_preds = preds[np.array(p_labels) == 1]
    g1_labels = labels[np.array(p_labels) == 1]

    g0_preds = preds[np.array(p_labels) == 0]
    g0_labels = labels[np.array(p_labels) == 0]

    tn0, fp0, fn0, tp0 = confusion_matrix(g0_labels, g0_preds).ravel()
    TPR0 = tp0/(fn0+tp0)
    TNR0 = tn0/(fp0+tn0)

    tn1, fp1, fn1, tp1 = confusion_matrix(g1_labels, g1_preds).ravel()
    TPR1 = tp1/(fn1+tp1)
    TNR1 = tn1/(tn1+fp1)
    
    acc_0 = accuracy_score(g0_preds, g0_labels)
    acc_1 = accuracy_score(g1_preds, g1_labels)

    if not silence:
        print("Accuracy 0: {}".format(acc_0))
        print("Accuracy 1: {}".format(acc_1))

        print("TPR 0: {}".format(TPR0))
        print("TPR 1: {}".format(TPR1))

        print("TNR 0: {}".format(TNR0))
        print("TNR 1: {}".format(TNR1))

        print("TPR gap: {}".format(TPR0-TPR1))
        print("TNR gap: {}".format(TNR0-TNR1))
    return {"Accuracy_0": acc_0,
            "Accuracy_1":acc_1,
            "TPR_0":TPR0,
            "TPR_1":TPR1,
            "TNR_0":TNR0,
            "TNR_1":TNR1,
            "TPR_gap":(TPR0-TPR1),
            "TNR_gap":(TNR0-TNR1)}

def leakage_evaluation(model, 
                    adv_level, 
                    training_generator,
                    validation_generator,
                    test_generator,
                    device):
    model.eval()
    model.adv_level = adv_level

    train_hidden = []
    train_labels = []
    train_private_labels = []

    for batch in training_generator:
        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        train_labels += list(tags.cpu().numpy() )
        train_private_labels += list(p_tags.cpu().numpy())
        
        text = text.to(device)

        hidden_state = model.hidden(text)
        train_hidden.append(hidden_state.detach().cpu().numpy())
    train_hidden = np.concatenate(train_hidden, 0)

    dev_hidden = []
    dev_labels = []
    dev_private_labels = []

    for batch in validation_generator:
        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        dev_labels += list(tags.cpu().numpy() )
        dev_private_labels += list(p_tags.cpu().numpy())
        
        text = text.to(device)

        hidden_state = model.hidden(text)
        dev_hidden.append(hidden_state.detach().cpu().numpy())
    dev_hidden = np.concatenate(dev_hidden, 0)

    test_hidden = []
    test_labels = []
    test_private_labels = []

    for batch in test_generator:
        text = batch[0]
        tags = batch[1]
        p_tags = batch[2]

        test_labels += list(tags.cpu().numpy() )
        test_private_labels += list(p_tags.cpu().numpy())
        
        text = text.to(device)

        hidden_state = model.hidden(text)
        test_hidden.append(hidden_state.detach().cpu().numpy())
    test_hidden = np.concatenate(test_hidden, 0)


    biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
    # biased_classifier = MLPClassifier(max_iter=50, batch_size=1024)
    # biased_classifier.fit(train_hidden, train_private_labels)
    biased_classifier.fit(dev_hidden, dev_private_labels)
    # dev_leakage = biased_classifier.score(dev_hidden, dev_private_labels)
    dev_leakage = biased_classifier.score(dev_hidden, dev_private_labels)
    test_leakage = biased_classifier.score(test_hidden, test_private_labels)
    print("Dev Accuracy: {}".format(dev_leakage))
    print("Test Accuracy: {}".format(test_leakage))