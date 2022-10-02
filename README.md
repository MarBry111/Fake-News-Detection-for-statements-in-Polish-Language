

# Fake-News-Detection-for-Social-Media-Posts-in-Polish-Language
Master thesis repository for thesis topic done at MINI faculty at Warsaw Universit of Technology (WUT). 
## Datasets
### Usage of existing labeled datasets
The idea is simple - fact-checking websites can provide "claim" with label as "fake news" or "truth" or twitter id with verification, then this data has to be scrapped/downloaded and the dataset is ready. 

- Data from demagog_nlp_cz - http://nlp.kiv.zcu.cz/research/fact-checking 
- Data from twitter_pl - https://zenodo.org/record/4319813 
- Data from getting_real_about_fake_news - https://www.kaggle.com/datasets/mrisdal/fake-news
- Data from oko.press - data, courtesy of OKO.press team (https://oko.press/) 
- Data from liar - https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

### Creation of weak supervised dataset
Idea behind this approach was inspired by paper [Weakly Supervised Learning for Fake News Detection on Twitter](https://ieeexplore.ieee.org/document/8508520) where authors created training dataset by assuming that all posts from trustworthy page is truth and from troll website is fake. As the extension of this method could be used some metrics from [Computational Propaganda in  Poland:  False Amplifiers and the Digital Public  Sphere](https://blogs.oii.ox.ac.uk/politicalbots/wp-content/uploads/sites/89/2017/06/Comprop-Poland.pdf), where various methods have been used to identify troll accounts.

## Methods
### Usage of Polish (benchmark)
First step was to recreate methods with data used in [Machine Learning Approach to Fact-Checking  in West Slavic Languages](https://aclanthology.org/R19-1113.pdf) paper and compare with my approach (the one used in paper was Ngrams with logistic regression). 
The dataset of polish claims has been used - only `TRUE` and `FALSE` statements (respectively 1761 and 648 records) and randomly undersampling them to get balanced dataset. Due to the lack of data approach used in this experiment was to evaluate by using 10-fold cross-validation (CV) by randomly splitting data and by using Latent Dirichlet Allocation (LDA) topic modelling to assign each of the statements into different topic (inspired by [Capturing the Style of Fake News](https://ojs.aaai.org//index.php/AAAI/article/view/5386)).  After obtaining test dataset for given fold the correlation between label and each of the features has been calculated and only data with values higher than absolute value greater than 0.05 or 0.01 were used.
Metrics to evaluate goodness of given model: accuracy and f1-score.

Approaches used (for each of them the Logistic Regression has been used with C=1 and L2 regularization) for random and based on topic 10-fold CV:

- usage of Ngrams of words,
- usage of features extracted from text (number of words, sentiment, etc.),
- usage of Ngrams of Part Of Speech (POS) tags,
- combined features and Ngrams od POS tags.

Expected results were that for random split there should be smaller difference between results of Ngrams of words and Ngrams of POS approach but in case of topic split (simulating real word situation for example: learning our model on fake-news about COVID and using it for classification of statements about war in Ukraine) the approaches relying on style of the the statement should get better performance. 

### Usage of Polish (demagog + oko.press)
The next step was using more data: scrapped from demagog website and shared with by by oko.press to run experiments compared to the ones in previous section. The pros from this approach would be having more reliable outcomes due to the dataset combined of  around 6000 records with balanced two classes.
Extra features has been tested - experimenting a bit with embeddings with usage of embeddings (Word2Vec, Doc2Vec, Transformers approaches).

### Training on different languages and testing on Polish
After reaching some more relatable results for more polish data, I wanted to test if using different languages for training purposes (features relying on style of the the statement) could give better results due to the greater amount of data.
Here the training set was combined with non polish data and test set consisted only polish records.

### Use twitter data (train on different languages and test on Polish)
As the last step the approach , after verifying for short statements/claims the goodness of different approaches and testing possibilities of using different languages as training dataset, obtained knowledge was used to train model on twitter data (mostly English) and tested this approach on Polish dataset.
