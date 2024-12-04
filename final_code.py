from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

'''
This file represents the final submission. It is the random forest model running on the selected final tuned
hyperparameters. This is Random Forest Classifcation with decades and represents the final model, with the final parameters.
'''

# Load the data
data = pd.read_csv('dataset/YearPredictionMSD.txt', delimiter=',')  # Adjust delimiter if necessary
print(data.head())

# Convert year to decade
# This results in MUCH better performance than using the year directly with the random forest classifier model
data['Decade'] = (data.iloc[:, 0] // 10) * 10

# Split the data
# This is following the recommended split provided with the dataset itself
train_data = data.iloc[:463715]
test_data = data.iloc[-51630:]

# Separate features and target for the training data
X_train = train_data.drop(columns=['Decade', train_data.columns[0]])  # Exclude the Year and Decade columns
y_train = train_data['Decade']    # The Decade column

# Separate features and target for the test data
X_test = test_data.drop(columns=['Decade', test_data.columns[0]])    # Exclude the Year and Decade columns
y_test = test_data['Decade']      # The Decade column

# Initialize the RandomForestClassifier with the ideal hyperparameters
rf_clf = RandomForestClassifier(n_estimators = 300, min_samples_split = 5, min_samples_leaf = 1, max_depth = 30, bootstrap = False)

# Fit the model on the training data
rf_clf.fit(X_train, y_train)

# Predict on the training set
y_train_pred = rf_clf.predict(X_train)

# Predict on the test set
y_test_pred = rf_clf.predict(X_test)

# Evaluate the model on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Training Accuracy: {train_accuracy}')
print('Training Classification Report:')
print(classification_report(y_train, y_train_pred))

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy}')
print('Test Classification Report:')
print(classification_report(y_test, y_test_pred))

# Plot the confusion matrix for the test set
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
# Save the confusion matrix plot
plt.savefig('confusion_matrix.png')

# Save the accuracies to a file
with open('model_accuracies.txt', 'w') as f:
    f.write(f'Training Accuracy: {train_accuracy}\n')
    f.write(f'Test Accuracy: {test_accuracy}\n')

    f.write('\n')
    
    f.write('Training Classification Report:\n')
    f.write('Test Classification Report:\n')
    f.write(classification_report(y_test, y_test_pred))
    f.write(classification_report(y_train, y_train_pred))


# Show the plot
plt.show()