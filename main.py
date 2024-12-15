from data_preprocessing import load_and_preprocess_data
from model_training import train_ml_models, build_and_train_dl_model
from evaluation import return_score, plot_confusion_matrix

# Load data
df = load_and_preprocess_data("train 2.csv")

# Split data
x_train, x_test, y_train, y_test = train_test_split(df["tweet_preprocessed"], df["class"], test_size=0.1, random_state=8)

history = build_and_train_dl_modeltrain_ml_models(x_train, y_train)

# Evaluate
y_pred = sgd.predict(x_test)
scores = return_score(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred)
plot_loss_and_accuracy(history)