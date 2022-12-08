# Fake News Detection for statements in Polish Language

Master thesis repository for thesis topic done at MINI faculty at Warsaw Universit of Technology (WUT). 

---

- [Datasets](#datasets)
  * [Usage of existing labeled datasets](#usage-of-existing-labeled-datasets)
  * [Data augmentation](#data-augmentation)
  * [Creation of weak supervised dataset](#creation-of-weak-supervised-dataset)
- [Methods](#methods)
  * [Usage of Polish (benchmark)](#usage-of-polish--benchmark-)
  * [Usage of Polish (demagog + oko.press)](#usage-of-polish--demagog---okopress-)
  * [Training on different languages and testing on Polish](#training-on-different-languages-and-testing-on-polish)
  * [Enhancing embeddings](#enhancing-embeddings)
    + [Triplet loss](#triplet-loss)
    + [Intermediate layer embeddings](#intermediate-layer-embeddings)
    + [Fine-tuning HerBERT](#fine-tuning-herbert)
    + [Application of StyloMetrix](#application-of-stylometrix)
  * [Use twitter data (train on different languages and test on Polish)](#use-twitter-data--train-on-different-languages-and-test-on-polish-)
- [Results](#results)
  * [Usage of Polish (benchmark)](#usage-of-polish--benchmark--1)
  * [Usage of Polish (demagog + oko.press)](#usage-of-polish--demagog---okopress--1)
    + [Results of logistic regression for different sets of features](#results-of-logistic-regression-for-different-sets-of-features)
    + [Results of transformers approach](#results-of-transformers-approach)
  * [Training on different languages and testing on Polish](#training-on-different-languages-and-testing-on-polish-1)
    + [Training](#training)
    + [Validation](#validation)
    + [Results of transformers approach](#results-of-transformers-approach-1)
  * [Modification of transformers embeddings](#modification-of-transformers-embeddings)
    + [Triplet loss (polish dataset)](#triplet-loss--polish-dataset-)
    + [StyloMetrix (polish dataset)](#stylometrix--polish-dataset-)
    + [Results](#results-1)

## Datasets
### Usage of existing labeled datasets
The idea is simple - fact-checking websites can provide "claim" with label as "fake news" or "truth" or twitter id with verification, then this data has to be scrapped/downloaded and the dataset is ready. 

- Data from demagog_nlp_cz - http://nlp.kiv.zcu.cz/research/fact-checking 
- Data from twitter_pl - https://zenodo.org/record/4319813 
- Data from getting_real_about_fake_news - https://www.kaggle.com/datasets/mrisdal/fake-news
- Data from oko.press - data, courtesy of OKO.press team (https://oko.press/) 
- Data from liar - https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- Data from politifact - https://www.kaggle.com/datasets/shivkumarganesh/politifact-factcheck-data

### Data augmentation
Augmentation in case of images is a easy task but how to deal with it in case of sequence where order matters? The idea behind this approach is to use model with dropout as augmentation - embeddings for the same sentence will be different due to usage of dropout (inspired by [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)).

### Creation of weak supervised dataset
Idea behind this approach was inspired by paper [Weakly Supervised Learning for Fake News Detection on Twitter](https://ieeexplore.ieee.org/document/8508520) where authors created training dataset by assuming that all posts from trustworthy page is truth and from troll website is fake. As the extension of this method could be used some metrics from [Computational Propaganda in  Poland:  False Amplifiers and the Digital Public  Sphere](https://blogs.oii.ox.ac.uk/politicalbots/wp-content/uploads/sites/89/2017/06/Comprop-Poland.pdf), where various methods have been used to identify troll accounts.

## Methods
### Usage of Polish (benchmark)
First step was to recreate methods with data used in [Machine Learning Approach to Fact-Checking  in West Slavic Languages](https://aclanthology.org/R19-1113.pdf) paper and compare with my approach (the one used in paper was Ngrams with logistic regression). 
The dataset of polish claims has been used - only `TRUE` and `FALSE` statements (respectively 1761 and 648 records) and randomly undersampling them to get balanced dataset. Due to the lack of data approach used in this experiment was to evaluate by using 10-fold cross-validation (CV) by randomly splitting data and by using Latent Dirichlet Allocation (LDA) topic modelling to assign each of the statements into different topic (inspired by [Capturing the Style of Fake News](https://ojs.aaai.org//index.php/AAAI/article/view/5386) - it was also inspiration for some of the features, along with [Identifying Different Writing Styles in a Document Intrinsically Using Stylometric Analysis](https://github.com/Hassaan-Elahi/Writing-Styles-Classification-Using-Stylometric-Analysis)).  After obtaining test dataset for given fold the correlation between label and each of the features has been calculated and only data with values higher than absolute value greater than 0.05 or 0.01 were used.
Metrics to evaluate goodness of given model: accuracy and f1-score.

Approaches used (for each of them the Logistic Regression has been used with C=1 and L2 regularization) for random and based on topic 10-fold CV:

- usage of Ngrams of words,
- usage of features extracted from text (number of words, sentiment, etc.),
- usage of Ngrams of Part Of Speech (POS) tags,
- combined features and Ngrams od POS tags.

Expected results were that for random split there should be smaller difference between results of Ngrams of words and Ngrams of POS approach but in case of topic split (simulating real word situation for example: learning our model on fake-news about COVID and using it for classification of statements about war in Ukraine) the approaches relying on style of the the statement should get better performance. 

### Usage of Polish (demagog + oko.press)
The next step was using more data: scrapped from demagog website and shared with by by oko.press to run experiments compared to the ones in previous section. The pros from this approach would be having more reliable outcomes due to the dataset combined of  around 6000 records with balanced two classes.

Extra features has been tested - experimenting a bit with embeddings with usage of Word2Vec embeddings averaged using Tf-Idf values of each of the words as weight. Also approach with usage of training dataset to train Doc2Vec embeddings has been used - which omits the need of averaging the vectors of words over statement. Finally the embeddings from [HerBERT model](https://huggingface.co/allegro/herbert-large-cased) were used as source of features for logistic regression model (last hidden state).

Transformers like models with help of [adapters framework](https://docs.adapterhub.ml/index.html) were used to see the possibilities which could be achieved using more sophisticated methods and spending more time on optimizing hyper-parameters of models. The HerBERT model was used as the source of embeddings and as the base for model, the downside of this solution was  the fact that usage of HerBERT produced 1024 features so one more approach has been used - applying on the top of HerBERT the PCA decreasing the dimensionality from 1024 to 100.
<!--
### Training on different languages and testing on Polish
After reaching some more relatable results for more polish data, I wanted to test if using different languages for training purposes (features relying on style of the the statement or numerical embeddings) could give better results due to the greater amount of data.
Here the training set was combined with non polish data and test set consisted only polish records (the one from previous section), the [demagog dataset for czech an slovak](http://nlp.kiv.zcu.cz/research/fact-checking) and [polifact english](https://www.kaggle.com/datasets/shivkumarganesh/politifact-factcheck-data) were used. 

For POS tagging the polyglot (for Czech and Slovak - using Czech setting) and spacy (for English) were used, for creation of words embeddings (averaged over the claim/statement) the [LaBSE model](https://huggingface.co/sentence-transformers/LaBSE) and [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) were used, the embeddings of original size 768 were transformed to 100 features using PCA technique.

At the end the [SlavicBERT](https://huggingface.co/DeepPavlov/bert-base-bg-cs-pl-ru-cased) was used with fully transformer approach (using adapters framework mentioned in previous section).

Whole dataset contained 38'465 records - 25'033 of and `TRUE` and 13'432 of `FALSE` so the undersampling technique was used to obtain balanced dataset of size 26'864.

During cross-validation given models were compared (vanilla variants):

- Logistic Regression with value of C=1,
- Gaussian Naive Bayes,
- K Nearest Neighbors with K=$\sqrt N$ (where N-number of point in training set, K=163),
- Random Forest,
- Support Vector Machine,
- XGboost,
- Voting model created with the previous ones.
-->

### Enhancing embeddings

The first step was usage of the HerBERT model for polish data (performing 10-fold cross-validation) and then SlavicBERT (training model on slavic data and testing on polish one) in case of most promising approach which could be applied to many languages. 

#### Triplet loss
The idea of modification embeddings to get the ones which will maximize the distances between desired classes was performed applying Triplet Loss for siamese network.

The input to the model was the last hidden state of the BERT like model and then some simple fully-connected layer will be optimized using Triplet Loss approach producing at the end 100 features used by logistic regression.

#### Intermediate layer embeddings
Outputs of intermediate layers were used as source of embeddings for final model, starting with using one at the time and ending with linear combination of each of the layers input using attention.

The main focus of usage of multilayer attention is to decide which hidden state produces the best representation of embeddings for given task, not only using the last layer ones.

#### Fine-tuning HerBERT
HerBERT model was fine-tuned for the task of classification fake news on training set and then the embeddings obtained in this manner were used.

#### Application of StyloMetrix
Values obtained with [StyloMetrix](https://github.com/ZILiAT-NASK/StyloMetrix) were used to improve embeddings of HerBERT.

### Use twitter data (train on different languages and test on Polish)
As the last step the approach, after verifying for short statements/claims the goodness of different approaches and testing possibilities of using different languages as training dataset, obtained knowledge was used to train model on twitter data (mostly English) and tested this approach on Polish dataset.

This step could be only performed if previous gave good.

## Results
The random seed was used to obtain comparable results.

### Usage of Polish (benchmark)
Results of logistic regression for different sets of features

Topic and random 10-fold splits results
|      embeddings      | accuracy (topic)| f1score (topic)| accuracy (random)| f1score (random)|
|:--------------------:|:------------:|:------------:|:------------:|:------------:|
| Ngrams of words    | 0.515+-0.034 | 0.501+-0.076 | 0.528+-0.044 | 0.508+-0.056 |
|  features          | 0.509+-0.032 | 0.454+-0.067 | 0.522+-0.029 | 0.466+-0.051 |
| Ngrams of POS tags |**0.561+-0.049**|**0.558+-0.049**|**0.563+-0.033**|**0.559+-0.046**|
|Ngrams POS + features|0.545+-0.059 | 0.549+-0.060 | 0.544+-0.043 | 0.540+-0.055 |

[//]: # (| Wor2vec embeddings | 0.519+-0.039 | 0.527+-0.047 | 0.516+-0.023 | 0.519+-0.020 |)

In both splits usage of Ngrams of POS tags gave best results, what is more the difference between accuracy for between Ngrams of words nad POS for topic approach gave greater difference than in case of random split (as expected).

### Usage of Polish (demagog + oko.press)
#### Results of logistic regression for different sets of features
Topic and random 10-fold splits results
|       embeddings     | accuracy (topic)| f1score (topic)| accuracy (random)| f1score (random)|
|:--------------------:|:------------:|:------------:|:------------:|:---------:|
| Ngrams of words     | 0.538+-0.058 | 0.099+-0.065 | 0.601+-0.015 | 0.470+-0.019|
|  features           | 0.529+-0.038 | 0.345+-0.063 | 0.539+-0.017 | 0.381+-0.024|
| Ngrams of POS tags  | 0.620+-0.030 | 0.504+-0.075 | 0.624+-0.015 | 0.536+-0.026|
| Ngrams POS + features| 0.623+-0.026 | 0.529+-0.058 | 0.625+-0.014| 0.554+-0.023|
| Wor2vec embeddings | 0.625+-0.022 | 0.574+-0.065 | 0.629+-0.019 | 0.593+-0.026 |
| Dov2vec embeddings | 0.598+-0.021 | 0.548+-0.040| 0.605+-0.019 | 0.566+-0.028 |
| HerBERT embeddings | **0.695+-0.017** | **0.675+-0.035** | 0.694+-0.016 | 0.680+-0.022 |
| HerBERT embeddings + PCA | 0.689+-0.013 | 0.664+-0.051 | **0.695+-0.009** | **0.680+-0.017** |

After using more data (5 times more) results showed that usage of Ngrams of POS tags, extra features and word2vec embeddings averaged using Tf-Idf values can give the best results, right after usage of the last hidden state of the HerBERT model as embeddings, in case of topic split of the data (more similar to real world scenario) and in case of random split (here the HerBERT and PCA approach reached even higher values of metrics or lower standard deviation of them).

After increasing the size of training dataset the embeddings obtained using last hidden state of HerBERT model yield the best result, then the ones extracted using word2vec averaged using Tf-Idf values. In case of "benchmark" dataset, the sizes of training sets could be to small to obtain any relevant embeddings.

The downside of the HerBERT approach is the fact that number of features is high (1024) comparing to number of examples in dataset (~6500), but after applying method of decreasing the dimensions to (arbitrary chosen) 100 using PCA comparable results were obtained.


#### Results of transformers approach
Topic and random 10-fold splits results
|   model  | accuracy (topic)|f1score (topic)|accuracy (random)|f1score (random)|
|:--------:|:------------:|:------------:|:------------:|:------------:|
| Transformers (HerBERT) | 0.710+-0.014 | 0.690+-0.035 | 0.719+-0.016 | 0.707+-0.016

Comparison of results obtained with adapters approach shows that using more sophisticated methods the results obtained could reach above values of 70% of accuracy which shows that even having so small dataset the results obtained could start looking more "acceptable". What is more interesting using only last hidden state of HerBERT as input for logistic regression gave comparable results.


<!--
### Training on different languages and testing on Polish
#### Training
Results of random 5-fold cross-validation for POS Ngrams and embeddings
|     model  |accuracy (pos)| f1score (pos)|accuracy (emb)| f1score (emb)| 
|:----------:|:------------:|:------------:|:------------:|:------------:|
| log reg    |  0.68+-0.19  |  0.56+-0.36  |  0.62+-0.12  |  0.54+-0.29  |
| gaussian nb|**0.69+-0.20**|**0.57+-0.36**|  0.63+-0.13  |  0.56+-0.27  |
| knn        |  0.66+-0.17  | 0.56+-0.34   |  0.60+-0.10  |**0.59+-0.20**|
| rand forest|**0.69+-0.20**|**0.57+-0.36**|**0.66+-0.16**|  0.59+-0.28  |
| svm        |  0.68+-0.19  |  0.56+-0.36  |  0.62+-0.11  |  0.53+-0.29  |
| xgboost    |  0.66+-0.15  |  0.55+-0.32  |  0.63+-0.11  |  0.57+-0.25  |
|voting model|**0.69+-0.20**|**0.57+-0.37**|  0.65+-0.15  |  0.56+-0.31  |
 
Increase of the standard deviation can be seen comparing to the results for polish-polish dataset, so even obtaining higher values doesn't say much due to high uncertainty of the result. The given results were not surprising due to usage of 3 languages and trying to map them to the common space of embeddings, using POS tagging or language agnostic deep learning model.

#### Validation
Results of voting model for polish dataset.
|     embeddings    |accuracy| f1score|
|:-----------------:|:------:|:------:|
| Ngrams of POS tags|  0.53  | 0.05   | 
| Embeddings + PCA  |  0.51  | 0.56   |  

Based on those it seems that usage of multi-language approach for training model returns worse results than using only one language. It could be driven by the fact that construction of sentences and speaking/writing rules are different for different languages and it is hard to use the knowledge of style of the fake-new extracted from one language to classify examples from the another one.

#### Results of transformers approach
|     model         |accuracy| f1score|
|:-----------------:|:------:|:------:|
| SlavicBERT | 0.45  | 0.60   | 

Usage of more sophisticated methods and only Slavic data doesn't yield any better results that the approach using Slavic and English data for training purposes. 
-->


### Modification of transformers embeddings
After obtaining the enhanced embeddings, each of them was tested using logistic regression.

#### Triplet loss (polish dataset)
Embeddings were obtained using HerBERT model with addition of simple fully-connected network at the end, which was optimized using Triplet Loss.

Networks was trained for 1000 epochs, with learning rate 1e-4 and L2 penalty (weight_decay) 1e-5. The version of the model which obtained the best results on validation set has been chosen and the obtained embeddings were used as features in logistic regression model.

There was no real improvement observed - but on the other hand the number of features was decreased from 1024 into 100, with better results than obtained with usage of PCA.

#### Triplet loss (augmented polish dataset)
Here the HerBERT model embeddings were obtained by keeping dropout layers activated and producing 10 embeddings for each statement, where model was optimized using Triplet Loss in the next steps. 

Despite 10 times bigger training dataset for Triplet loss part the whole experiment was performed in the same manner a the previous one, using in logistic regression embeddings from HerBERT with dropout deactivated. 

The great decrease in accuracy and f1 score was observed in case of topic driven cross validation, and only slightly decreased metric for random splits. 

####  StyloMetrix (polish dataset)

Adding the StyloMetrix values (89 of them) to the end of HerBERT embedding and improving size from 1024 to ~1100 gave the best results. Model was able to obtain accuracy higher than for Adapters approach described in previous section (in case of topic driven approach).

What is interesting usage of only StyloMetrix features gave quiet good results, better than features obtained with usage of spaCy (Ngrams POS, text features), Word2Vec and Doc2Vec but slightly worse than ones obtained with usage pure HerBERT.

The main disadvantage of this solution is fact that it can't be used for data different than English and Polish - due to "the tool" limitations.

#### Results
|   embeddings  | accuracy (topic)|f1score (topic)|accuracy (random)|f1score (random)|
|:--------:|:------------:|:------------:|:------------:|:------------:|
| HerBERT embeddings | 0.695+-0.017 | 0.675+-0.035 | 0.694+-0.016 | 0.680+-0.022 |
| HerBERT embeddings <br> + PCA | 0.689+-0.013 | 0.664+-0.051 | 0.695+-0.009 | 0.680+-0.017 |
| HerBERT embeddings <br> + Triplet Loss | 0.694+-0.031 | 0.678+-0.028 |  0.691+-0.062 | 0.677+-0.041 |
| HerBERT embeddings augmented <br> + Triplet Loss |  0.632+-0.086 | 0.643+-0.058 | 0.667+-0.051 | 0.667+-0.035 |
| StyloMetrix | 0.646+-0.003 | 0.617+-0.016 |  0.645+-0.002 | 0.617+-0.004 |
| HerBERT embeddings <br> + StyloMetrix | **0.778+-0.003** | **0.769+-0.004** |  **0.705+-0.002** | **0.692+-0.002** |



