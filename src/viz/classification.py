# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay, plot_confusion_matrix



def printout_metrics(y, y_hat, class_names=None):
    # Make prediction report
    clrep = classification_report(y,y_hat, output_dict=True)
    clrep = pd.DataFrame(clrep).transpose()
    ix = [int(i_) for i_ in clrep.index.values if i_.isdigit()]
    clrep['class_name'] = np.concatenate( (class_names[ix], ['', '', '']) )

    # Arrange results table
    clrep_bot = clrep.copy()
    print( clrep_bot.iloc[:25].sort_values(by="f1-score", ascending=True)[:10].append(clrep_bot.iloc[25:]) )
    return clrep


def printout_confusion(y, y_hat, class_names=None):
    cm = confusion_matrix(y, y_hat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot()

    fig = disp.ax_.get_figure()
    fig.axes[0].set_xticklabels(class_names, rotation='vertical')
    fig.set_figwidth(10)
    fig.set_figheight(10)
    return fig


def printout_graph_f1score_vs_nbinst(classif_report):
    hf = plt.figure()
    ax = hf.add_subplot(1, 1, 1)
    ax.plot( classif_report['support'],
            classif_report['f1-score'],
            '*r')
    ax.set_xlabel('Nb of instances')
    ax.set_ylabel('Correctness rate')

    ax.set_xscale('log')
    return hf, ax