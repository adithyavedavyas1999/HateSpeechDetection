# Hate Speech Detection

## Overview
This project focuses on detecting hate speech and offensive language in tweets using a combination of machine learning and deep learning models. The goal is to classify tweets into binary categories: hate/offensive and non-hate speech. The project demonstrates various techniques for data preprocessing, feature extraction, and model training while addressing challenges such as class imbalance.

The implementation involves:
	1.	Preprocessing and cleaning raw text data.
	2. 	Training and evaluating traditional machine learning models (e.g., Logistic Regression, Random Forest).
	3.	Building a deep learning model using an LSTM-based architecture for sequence analysis.
	4.	Visualizing model performance with metrics and confusion matrices.

The dataset is sourced from the Hate and Offensive Language Detection project on Kaggle. It contains tweets labeled into three categories:

	1.	Hate Speech: Tweets containing hate speech.
	2.	Offensive Language: Tweets that are offensive but not classified as hate speech.
	3.	Neither: Neutral tweets that are neither hate speech nor offensive.

Dataset Link:

Hate and Offensive Language Detection Dataset on Kaggle - https://www.kaggle.com/code/thechief28/hate-and-offensive-language-detection/input

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt```
2. Run the project:
    ```bash
    python main.py```