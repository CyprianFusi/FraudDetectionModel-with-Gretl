import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

def cap_curve(y_test, y_pred):
    
    total = len(y_test)             # length of the test data
    one_count = np.sum(y_test)      # Counting '1' labels in test data
    zero_count = total - one_count  # Counting '0' labels in test data
    lm = [y for _, y in sorted(zip(y_pred, y_test), reverse = True)]
    x = np.arange(0, total + 1)
    y = np.append([0], np.cumsum(lm))
    
    plt.plot([0, total], [0, one_count], c = 'b', linestyle = '--', label = 'Random Model')
    plt.plot(x, y, c = 'b', label = 'Trained Model', linewidth = 2)
    plt.plot([0, one_count, total], [0, one_count, one_count], c = 'grey', 
             label = 'Perfect Model', linewidth = 2)
    
    # Point where vertical line will cut trained model
    index = int((50*total / 100))

    ## 50% Verticcal line from x-axis
    plt.plot([index, index], [0, y[index]], c ='g', linestyle = '--')

    ## Horizontal line to y-axis from trained model
    plt.plot([0, index], [y[index], y[index]], c = 'g', linestyle = '--')
    class_1_observed = y[index] * 100 / max(y)
    plt.xlabel('Total observations', fontsize = 10)
    plt.ylabel('Class 1 observations', fontsize = 10)
    plt.title('Cumulative Accuracy Profile', fontsize = 10)
    plt.legend()
    plt.show()
    print(f'Percentage: {class_1_observed:.2f}%')
    print('''
    Interpret this percentage as follows:
    
    1. Less than 60%: Useless Model
    2. 60% - 70%: Poor Model
    3. 70% - 80%: Good Model
    4. 80% - 90%: Very Good Model
    5. More than 90%: Too Good to be True.\n
    In the fifth case, one must always check for overfitting.
    ''')
    
def auc_curve(y_test, y_pred):
    plt.plot([0,1], [0,1], 'r--')

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    label = 'Model AUC:' + ' {0:.2f}'.format(roc_auc)
    plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 2)
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.title('Receiver Operating Characteristic', fontsize = 14)
    plt.legend(loc = 'lower right', fontsize = 12)
    plt.show()
    
def feature_importance(model, X_train):    
    importance = pd.DataFrame({'Importance': model.feature_importances_*100}, index = X_train.columns)
    importance.sort_values('Importance', axis = 0, ascending = True).plot(kind = 'barh', color = 'r', )
    plt.show()