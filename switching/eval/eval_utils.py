import matplotlib
import matplotlib.cm
import numpy as np
import scipy.spatial as spatial


def micro_average(measuresList):
    microAverage = dict()
    eps = 1e-12

    TP = sum([dict_['TP'] for dict_ in measuresList])
    TN = sum([dict_['TN'] for dict_ in measuresList])
    FP = sum([dict_['FP'] for dict_ in measuresList])
    FN = sum([dict_['FN'] for dict_ in measuresList])

    # Accuracy
    microAverage['accuracy'] = (TP + TN) / (TP + FP + TN + FN)

    # Precision
    microAverage['precision'] = TP / (TP + FP + eps)

    # Specificity
    microAverage['specificity'] = TN / (FP + TN + eps)

    # Recall
    microAverage['recall'] = TP / (TP + FN + eps)

    # F-measure
    microAverage['f1'] = 2 * microAverage['precision'] * microAverage['recall'] / (
                microAverage['recall'] + microAverage['precision'])

    # Negative Predictive Value
    microAverage['npv'] = TN / (FN + TN + eps)

    # False Predictive Value
    microAverage['fpr'] = FP / (FP + TN + eps)

    print('Accuracy ', microAverage['accuracy'], '\n',
          'Precision', microAverage['precision'], '\n',
          'Recall', microAverage['recall'], '\n',
          'Specificity ', microAverage['specificity'], '\n',
          'F-measure', microAverage['f1'], '\n',
          'NPV', microAverage['npv'], '\n',
          'FPV', microAverage['fpr'], '\n')

    return microAverage


def macro_average(measuresList):
    macroAverage = dict()

    # Accuracy
    macroAverage['accuracy'] = np.mean([dict_['accuracy'] for dict_ in measuresList])

    # Precision
    macroAverage['precision'] = np.mean([dict_['precision'] for dict_ in measuresList])

    # Specificity
    macroAverage['specificity'] = np.mean([dict_['specificity'] for dict_ in measuresList])

    # Recall
    macroAverage['recall'] = np.mean([dict_['recall'] for dict_ in measuresList])

    # F-measure
    macroAverage['f1'] = np.mean([dict_['f1'] for dict_ in measuresList])

    # Negative Predictive Value
    macroAverage['npv'] = np.mean([dict_['npv'] for dict_ in measuresList])

    # False Predictive Value
    macroAverage['fpr'] = np.mean([dict_['fpr'] for dict_ in measuresList])

    print('Accuracy ', macroAverage['accuracy'], '\n',
          'Precision', macroAverage['precision'], '\n',
          'Recall', macroAverage['recall'], '\n',
          'Specificity ', macroAverage['specificity'], '\n',
          'F-measure', macroAverage['f1'], '\n',
          'NPV', macroAverage['npv'], '\n',
          'FPV', macroAverage['fpr'], '\n')

    return macroAverage


def compute_measures(list_, positiveClass=0):
    measures = dict()
    eps = 1e-12
    len_list = len(list_)
    # True positives TP : number of prediction that matches the GT
    TP = sum((list_[i][1] == positiveClass) and (list_[i][0]==positiveClass) for i in range(len_list))
    # True negatives TN
    TN = sum((list_[i][1] != positiveClass) and (list_[i][0]!=positiveClass) for i in range(len_list))
    # False positives FP
    FP = sum((list_[i][1] != positiveClass) and (list_[i][0]==positiveClass) for i in range(len_list))
    # False negatives FN
    FN = sum((list_[i][1] == positiveClass) and (list_[i][0]!= positiveClass) for i in range(len_list))
    print('TP ', TP, 'TN ', TN, 'FP', FP, 'FN', FN, 'Total', TP+TN+FP+FN)
    measures['TP'] = TP
    measures['TN'] = TN
    measures['FP'] = FP
    measures['FN'] = FN
    # Accuracy
    measures['accuracy'] = (TP+TN)/(TP+TN+FP+FN)
    # Precision
    measures['precision'] = TP/(TP+FP+eps)
    # Specificity
    measures['specificity']= TN / (FP + TN + eps)
    # Recall
    measures['recall'] = TP / (TP + FN + eps)
    # F-measure  dice score
    measures['f1'] = 2 * measures['precision'] * measures['recall'] / (measures['recall'] + measures['precision'] + eps)
    # Negative Predictive Value
    measures['npv'] = TN / (FN + TN + eps)
    # False Predictive Value
    measures['fpr'] = FP / (FP + TN + eps)
    return measures



def compute_measures_multiple(list_, positiveClass=0):
    measures = dict()
    eps = 1e-12
    len_list = len(list_)

    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len_list):
        cam_gt_label = [list_[i][1],list_[i][2],list_[i][3],list_[i][4],list_[i][5]]
        if cam_gt_label[positiveClass-1] == 1:
            if list_[i][0] == positiveClass:
                TP += 1
            else:
                FN += 1
        elif cam_gt_label[positiveClass-1] == 0:
            if list_[i][0] == positiveClass:
                FP += 1
            else:
                TN += 1

    print('TP ', TP, 'TN ', TN, 'FP', FP, 'FN', FN, 'Total', TP+TN+FP+FN)
    measures['TP'] = TP
    measures['TN'] = TN
    measures['FP'] = FP
    measures['FN'] = FN
    # Accuracy
    measures['accuracy'] = (TP+TN)/(TP+TN+FP+FN)
    # Precision
    measures['precision'] = TP/(TP+FP+eps)
    # Specificity
    measures['specificity']= TN / (FP + TN + eps)
    # Recall
    measures['recall'] = TP / (TP + FN + eps)
    # F-measure  dice score
    measures['f1'] = 2 * measures['precision'] * measures['recall'] / (measures['recall'] + measures['precision'] + eps)
    # Negative Predictive Value
    measures['npv'] = TN / (FN + TN + eps)
    # False Predictive Value
    measures['fpr'] = FP / (FP + TN + eps)
    return measures

def show_confusion_matrix(dict_, idx):
    from tabulate import tabulate
    print('Cam{} \'s confusion matrix '.format(idx))
    headers = ["-------", "Positive", "Negative"]
    table = [['   True  ', dict_['TP'], dict_['FN']],
             ['   False ', dict_['FP'], dict_['TN']],
             ]
    result = tabulate(table, headers, tablefmt="grid")
    print(result)


def calc_confusion_matrix(list_):
    multiclass = []
    for k in range(5):
        print('For class cam=', k+1)
        multiclass.append(compute_measures(list_, positiveClass=k+1))
    print('Macro-average')
    macro_average(multiclass)

    print('Micro-average')
    micro_average(multiclass)

    for idx in range(5):
        show_confusion_matrix(multiclass[idx], idx+1)

def calc_confusion_matrix_my(list_, show_matrix=True):
    multiclass = []
    for k in range(5):
        print('For class cam=', k+1)
        multiclass.append(compute_measures_multiple(list_, positiveClass=k+1))
    print('Macro-average')
    macro_average(multiclass)

    print('Micro-average')
    micro_average(multiclass)

    if show_matrix:
        for idx in range(5):
            show_confusion_matrix(multiclass[idx], idx+1)


def return_micro_precision_score(list_):
    multiclass = []
    for k in range(5):
        multiclass.append(compute_measures_multiple(list_, positiveClass=k + 1))
    eps = 1e-12
    TP = sum([dict_['TP'] for dict_ in multiclass])
    TN = sum([dict_['TN'] for dict_ in multiclass])
    FP = sum([dict_['FP'] for dict_ in multiclass])
    FN = sum([dict_['FN'] for dict_ in multiclass])
    # Precision
    microAverage_precision= TP / (TP + FP + eps)
    return microAverage_precision










