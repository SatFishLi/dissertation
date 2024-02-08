import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, \
    cohen_kappa_score, mean_squared_error

warnings.filterwarnings('ignore')
# Read the result file
files = ['decision_tree.npz', 'logistic_regression.npz', 'mlp.npz', 'rf.npz', 'svm.npz']
models = ['Decision Tree', 'Logistic Regression', 'MLP', 'Random Forest', 'SVM']

f1_scores_train = []
accuracies_train = []
recalls_train = []
precisions_train = []
kappas_train = []
rmses_train = []

f1_scores_test = []
accuracies_test = []
recalls_test = []
precisions_test = []
kappas_test = []
rmses_test = []

for index, file in enumerate(files):
    data = np.load('result/' + file)
    y_train = data['train_true']
    y_pred_train = data['train_pred']
    y_test = data['test_true']
    y_pred_test = data['test_pred']

    # Calculating Confusion Matrix Indicators - Training Set
    cm_train = confusion_matrix(y_train, y_pred_train)
    f1_score_train = f1_score(y_train, y_pred_train, average='weighted')
    accuracy_train = accuracy_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train, average='weighted')
    precision_train = precision_score(y_train, y_pred_train, average='weighted')
    kappa_train = cohen_kappa_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # Calculating Confusion Matrix Indicators - Test Set
    cm_test = confusion_matrix(y_test, y_pred_test)
    f1_score_test = f1_score(y_test, y_pred_test, average='weighted')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    precision_test = precision_score(y_test, y_pred_test, average='weighted')
    kappa_test = cohen_kappa_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f'Model {models[index]}:')
    print('Train F1',f1_score_train)
    print('Train ACC',accuracy_train)
    print('Train Recall',recall_train)
    print('Train Precision',precision_train)
    print('Train Kappa',kappa_train)
    print('Train RMSE',rmse_train)

    print('Test F1',f1_score_test)
    print('Test ACC',accuracy_test)
    print('Test Recall',recall_test)
    print('Test Precision',precision_test)
    print('Test Kappa',kappa_test)
    print('Test RMSE',rmse_test)

    print()

    # Add to the appropriate list
    f1_scores_train.append(f1_score_train)
    accuracies_train.append(accuracy_train)
    recalls_train.append(recall_train)
    precisions_train.append(precision_train)
    kappas_train.append(kappa_train)
    rmses_train.append(rmse_train)

    f1_scores_test.append(f1_score_test)
    accuracies_test.append(accuracy_test)
    recalls_test.append(recall_test)
    precisions_test.append(precision_test)
    kappas_test.append(kappa_test)
    rmses_test.append(rmse_test)

    plt.figure(dpi=300)
    sns.heatmap(cm_train,annot=True,fmt='d',cmap='Blues')
    plt.title(f'{models[index]} Train Confusion Martix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'figure/Train {models[index]}',dpi=600,bbox_inches='tight')
    # plt.show()

    plt.figure(dpi=300)
    sns.heatmap(cm_test,annot=True,fmt='d',cmap='Blues')
    plt.title(f'{models[index]} Test Confusion Martix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'figure/Test {models[index]}',dpi=600,bbox_inches='tight')
    # plt.show()

# Plotting Histograms - Training Set
labels = ['F1 Score', 'Accuracy', 'Recall', 'Precision', 'Kappa', 'RMSE']
x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
for i in range(len(models)):
    ax.bar(x + (i - len(models) / 2) * width,
           [f1_scores_train[i], accuracies_train[i], recalls_train[i], precisions_train[i],
            kappas_train[i], rmses_train[i]], width, label=models[i])

ax.set_ylabel('Scores')
ax.set_title('Evaluation Metrics - Training Set')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(ls='--', alpha=0.5)
plt.savefig(f'figure/Train Bar', dpi=600, bbox_inches='tight')
plt.show()

# Plotting Bar Charts - Test Set
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
for i in range(len(models)):
    ax.bar(x + (i - len(models) / 2) * width,
           [f1_scores_test[i], accuracies_test[i], recalls_test[i], precisions_test[i],
            kappas_test[i], rmses_test[i]], width, label=models[i])

ax.set_ylabel('Scores')
ax.set_title('Evaluation Metrics - Test Set')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(ls='--', alpha=0.5)
plt.savefig(f'figure/Test Bar', dpi=600, bbox_inches='tight')
plt.show()