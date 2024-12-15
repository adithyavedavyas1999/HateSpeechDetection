# Import required libraries for machine learning and deep learning
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import keras
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Train and validate traditional machine learning models
def train_ml_models(x_train, y_train):
    """
    Trains multiple machine learning models and evaluates them using stratified K-fold validation.
    
    Args:
    x_train: Training feature data (tweets).
    y_train: Training target labels.
    
    Returns:
    None. Prints model performance metrics for each model.
    """
    # List of machine learning models to train
    dummy = DummyClassifier()
    logistic = LogisticRegression(max_iter=1_000, random_state=8)
    svc = LinearSVC(max_iter=1_000, random_state=8)
    sgd = SGDClassifier()
    knn = KNeighborsClassifier()
    bayes = MultinomialNB()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    xgb = XGBClassifier()
    models = [
        dummy,
        logistic,
        sgd,
        dt,
        rf,
        xgb,
    ]
    
    # Loop through each model
    for model in models:
        model_name = model.__class__.__name__  # Get the model's name
        model_scores = []
        print(model_name)
        
        # Use Stratified K-Fold for cross-validation
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=8)
        for i, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):
            print(f"\tFold {i}")

            # Split data into training and validation sets for this fold
            x_train_count = x_train[train_index]
            y_train_count = y_train[train_index]
            x_val_count = x_train[val_index]
            y_val_count = y_train[val_index]

            # Oversample the minority class in training data to handle class imbalance
            ros = RandomOverSampler(random_state=42)
            x_train_count, y_train_count = ros.fit_resample(x_train_count.reshape(-1, 1), y_train_count)
            x_train_count = x_train_count.flatten()

            # Apply Count Vectorizer for text data (except for DummyClassifier)
            if model_name != "DummyClassifier":
                vectorizer = CountVectorizer()
                vectorizer.fit(x_train_count)
                x_train_count = vectorizer.transform(x_train_count)
                x_val_count = vectorizer.transform(x_val_count)

            # Train the model
            model.fit(x_train_count, y_train_count)

            # Predict on validation set
            y_pred = model.predict(x_val_count)

            # Evaluate model performance
            scores_this_fold = return_score(y_val_count, y_pred)
            model_scores.append(scores_this_fold)

        # Compute and print the average performance metrics across folds
        mean_acc = np.asarray([score['accuracy'] for score in model_scores]).mean()
        mean_f1 = np.asarray([score['f1'] for score in model_scores]).mean()
        mean_precision = np.asarray([score['precision'] for score in model_scores]).mean()
        mean_recall = np.asarray([score['recall'] for score in model_scores]).mean()
        print(f"\tAcc: {mean_acc: .5f} | F1: {mean_f1: .5f} | Precision : {mean_precision: .5f} | Recall: {mean_recall: .5f}")


# Train deep learning models
def build_and_train_dl_model(X_train, Y_train, X_val, Y_val):
    """
    Builds, trains, and evaluates a deep learning model for text classification.
    
    Args:
    X_train: Training feature data (tweets).
    Y_train: Training target labels.
    X_val: Validation feature data.
    Y_val: Validation target labels.
    
    Returns:
    Training history object containing metrics across epochs.
    """
    # One-hot encode target labels
    Y_train = pd.get_dummies(Y_train)
    Y_val = pd.get_dummies(Y_val)
    
    max_words = 5000  # Maximum number of words to keep in the vocabulary
    max_len = 100     # Maximum sequence length for padding

    # Tokenizer for converting text to sequences
    token = Tokenizer(num_words=max_words, lower=True, split=' ')
    token.fit_on_texts(X_train)

    # Convert text to padded sequences
    Training_seq = token.texts_to_sequences(X_train)
    Training_pad = pad_sequences(Training_seq, maxlen=max_len, padding='post', truncating='post')
    Testing_seq = token.texts_to_sequences(X_val)
    Testing_pad = pad_sequences(Testing_seq, maxlen=max_len, padding='post', truncating='post')

    # Convert target labels to one-hot encoded format
    train_label_one = to_categorical(Y_train, num_classes=2)
    test_label_one = to_categorical(Y_val, num_classes=2)

    # Define the deep learning model architecture
    model = keras.models.Sequential([
        layers.Embedding(max_words, 32, input_length=max_len),
        layers.Bidirectional(layers.LSTM(16)),
        layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l1()),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  # Output layer with softmax activation
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
    model.summary()

    # Add early stopping and learning rate reduction callbacks
    es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
    lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=1)

    # Train the model
    history = model.fit(Training_pad, np.argmax(Y_train.values, axis=1),  # Convert one-hot to integer labels
                        validation_data=(Testing_pad, np.argmax(Y_val.values, axis=1)),
                        epochs=25,
                        verbose=1,
                        batch_size=32,
                        callbacks=[es, lr])
    
    return history