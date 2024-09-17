import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and preprocess the data
# input: filepath: str (path to the CSV file)
# output: tuple of X (features), y (target)
def load_and_preprocess_data(filepath):
    # TODO: Implement this function
    dataset = pd.read_csv(filepath)
    
    # Remove garbage values column if it exists
    if 'GarbageValues' in data.columns:
        data = data.drop(columns=['GarbageValues'])
    
    # Drop rows with missing values
    data = data.dropna()
    
    #Separate the features and target
    X = datadet.drop('Outcome',axis=1)
    Y = dataset['Outcome'] 
    
    return X, Y

# Split the data into training and testing sets
# input: 1) X: ndarray (features)
#        2) y: ndarray (target)
# output: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X, y):
    # TODO: Implement this function
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

    
   

# Create and train 2 MLP classifiers with different parameters
# input:  1) X_train: ndarray
#         2) y_train: ndarray
# output: tuple of models (model1, model2)
def create_model(X_train, y_train):
    # TODO: Implement this function
    # Create and train the first model
    model1 = MLPClassifier(hidden_layer_sizes=(128,64, 32), max_iter=3000, random_state=50, learning_rate='adaptive', activation='relu', solver='adam', alpha=0.0001)

    model1.fit(X_train, y_train)
    
    # Create and train the second model
    model2 = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=3000, random_state=50, learning_rate='constant', activation='tanh', solver='sgd', alpha=0.0005)
    model2.fit(X_train, y_train)
    
    return model1, model2
   

# Predict and evaluate the model
# input: 1) model: MLPClassifier after training
#        2) X_test: ndarray
#        3) y_test: ndarray
# output: tuple - accuracy, precision, recall, fscore, confusion_matrix
def predict_and_evaluate(model, X_test, y_test):
    # TODO: Implement this function
    # Predict the target values
    y_pred = model.predict(X_test)
    
    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    fscore = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    print ('accuracy:', accuracy)
    print ('precision:', precision)
    print ('recall:', recall)
    print ('fscore:', fscore)
    print ('confusion:', confusion)
    
    return accuracy, precision, recall, fscore, confusion