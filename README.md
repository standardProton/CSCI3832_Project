# CSCI 3832 README 

**Group Members:** Alex Mcdonald, Aaron Semones, Alex Ludwigson, Yufan Qian, Gabo Sambo

#Introduction

Malicious emails, such as phishing attacks or spam emails, are becoming increasingly prevalent. In this project we use 4 models (Transformer, Bert, Gpt2, and Logistic Regression) to try and compare how these models perform in spam email and phishing URL detection

# Links

Spam Dataset: https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification/data

Phishing URLs Dataset: https://www.kaggle.com/datasets/hammadjavaid/phishing-url-dataset-for-nlp-based-classification

Spam trained transformer: https://drive.google.com/file/d/1EJVmrpsFpFKZ6Pan-uDlldhvePwwqCGu/view?usp=drive_link

URL trained transformer: https://drive.google.com/file/d/1r6pTQE898piOXyyc2QHrt7nmKT2sfD4j/view?usp=sharing

Checkpoint of Bert model for spam:

https://drive.google.com/drive/folders/1FlyQnDYppXTEJqlmxvEVr1wqwsiZfHQf?usp=sharing


# Instructions

1. Our project uses 2 datasets, both are sourced from Kaggle. To recreate our results, start with the data_stats.ipynb file.

2. Pip install torch, transformers, pandas, numpy, nltk, tqdm, seaborn, and scikit-learn.

3. Create new directories titled “./datasets/SpamHam” and “./datasets/PhishingURLs”

4. In the data_stats.ipynb file, download the csv files from the 2 Kaggle links and put them in their respective directories.

5. Run each cell in the data_stats.ipynb notebook sequentially. This should create a train & test split on the spam dataset, and correct the format for the URLs csv.

6. If code loading datasets fails to run, change the relative pathing from “./REL_PATH_TO_FILE” to “../REL_PATH_TO_FILE” or vice versa

7. Create a directory named “./trained_models” and download the spam.pt, url.pt, and bert model from the links above (Must be logged into CU account). Place each inside./trained_models

8. Follow any further instructions in the markdown cells of each file

From there, it should have everything it needs to run the other notebook files. 

# Statement of Contributions

**Alex McDonald:** 

I applied exploratory data analysis on the 2 datasets to understand the nature of the data that we were working with (data_stats.ipynb). I then used pytorch to train a Spam and URL transformer model (transformer_train.ipynb). This involved picking hyperparameters and configuring the loss function to reflect the importance in the models’ recall scores, as well as developing an understanding of how to use pytorch’s encoder layers, positional encoding, and how to debug when the model fails to train. I then analyzed random samples of the incorrect inferences that the model made to speculate on possible steps that could improve the model’s performance (trained_transformer_error_analysis.ipynb).

**Alex Ludwigson:**

 I crafted the handmade dataset, using AI generated examples (One of the things that we sought to detect in our project proposal) and real examples from my personal email. I specifically looked for examples from my email that might throw the models off, such as password reset emails, which are commonly misclassified and end up in people's spam folders, and emails containing odd characters, such as emojis. I ran tests on all of the models to compare the results/performance on this dataset in the file handmadeEval.ipynb, which involved a lot of debugging in trying to figure out how to integrate this dataset with the other models code. For evaluation, I used both accuracy and F-scores. I also produced confusion matrices so we could look into the false positive rates, false negative rates, etc. Additionally, I also contributed to the logistic regression (logistic_regression.ipynb) document and added a validation/test split and re-worked the code that loaded the datasets.

**Aaron Semones:**

I wrote the entirety of the bert_finetine.ipynb. I first researched potential models to choose from on hugging face, and how to implement each. I then made a class to handle the training and evaluation of both gpt2 and bert and wrote a few helper functions to handle tokenization/statistics. I configured the models’ hyperparameters to work with my GPU which was a surprisingly long process. I trained, evaluated, and tested gpt2 and bert on the small dataset and the superior model (bert) on the dataset. Finally, I wrote the majority of the presentation slides.

**Yufan Qian:**

I planned to design and implement an LSTM model to do the classification work based on our two different datasets. The LSTM model could split the information from our datasets well and return the results when splitting has been finished, but it has a type mismatch error doing the next step and I feel like it might be caused by processing the information from string and storing the return value to an array. There is no exact result from LSTM when doing the classification emails, but theoretically, the LSTM model will have a better performance in classifying rich content than the normal emails because of its unique structure. We did not mention the LSTM model in our presentation since it had some bugs and did not run correctly.

**Gabo Sambo:**

I did the work on the baseline model doing the logistic regression model used in our project. My code handled the loading and preprocessing of both spam and phishing URLs datasets. I introduced the use of TfidfVectorizer to convert text data into feature vectors capping the features at 5000. I configured and trained the logistic regression model with balanced class weights to address class imbalances and ran multiple evaluations to assess the model's performance across different data splits, including our custom 'Homebrew' dataset. These evaluations focused on accuracy metrics, confusion matrices, and classification reports to provide a comprehensive overview of model performance to compare with the more complex models. (logistic_regression.ipynb)
