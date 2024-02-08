import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Due to personal habits, some of the notes and entries were originally in Chinese.
# Therefore, the following sections are used to make the text etc. display correctly
warnings.filterwarnings('ignore')
# Setting the Chinese font
plt.rcParams['font.family'] = 'SimHei'
# addressing the display of minus signs ('-') in Matplotlib plots.
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel('data/data.xlsx', index_col=0)  # Reading data from an Excel file

X = data.iloc[:, :8].values  # Extract the feature column as input X
Y = data.iloc[:, 8:9].values  # Extract the label column as output Y

scaler = MinMaxScaler()  # Create MinMaxScaler object for feature normalisation
encoder = LabelEncoder()  # Create LabelEncoder object for label encoding

x = scaler.fit_transform(X)  # normalised characteristic column
y = encoder.fit_transform(Y)  # Coded Label Column

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # Divide the training set and validation set

# Creating a Decision Tree Classifier Object
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(x_train, y_train)  # Fitting a classifier on the training set

# Prediction on the training set
train_predictions = classifier.predict(x_train)
# Prediction on the validation set
test_predictions = classifier.predict(x_test)

# Print the classification report on the training set
print("Training set classification report:")
print(classification_report(y_train, train_predictions))
print()

# Print the classification report on the validation set
print("Validation set classification report:")
print(classification_report(y_test, test_predictions))
print()

# Compute the confusion matrix on the training set
train_cm = confusion_matrix(y_train, train_predictions)
# Compute the confusion matrix on the validation set
test_cm = confusion_matrix(y_test, test_predictions)

# Visualising the confusion matrix on the training set
sns.heatmap(train_cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Training set confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('true')
plt.show()

# Visualising the confusion matrix on the validation set
sns.heatmap(test_cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('validation set confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('truw')
plt.show()

# Save the true and predicted labels of training and test sets to NPZ files
# Binary format, suitable for python processing analysis
np.savez('result/decision_tree.npz',
         train_true=y_train, train_pred=train_predictions,
         test_true=y_test, test_pred=test_predictions)


# Create a DataFrame of the training set
train_df = pd.DataFrame({
    'true_label': y_train.flatten(),
    'predicted_label': train_predictions.flatten()
})

# Create a DataFrame of the test set
test_df = pd.DataFrame({
    'true_label': y_test.flatten(),
    'predicted_label': test_predictions.flatten()
})

# DataFrame for merging training and test sets
df = pd.concat([train_df, test_df], ignore_index=True)

# Save as Excel file
df.to_excel('result/decision_tree.xlsx', index=False)