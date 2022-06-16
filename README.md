
# Fake-News-Detection-for-Social-Media-Posts-in-Polish-Language
Master thesis repository for thesis topic done at MINI faculty at Warsaw Universit of Technology (WUT). 
## Datasets
### Usage of existing labeled datasets
The idea is simple - fact-checking websites can provide "claim" with label as "fake news" or "truth" or twitter id with verification, then this data has to be scrapped/downloaded and the dataset is ready. 

- Data from demagog_nlp_cz - http://nlp.kiv.zcu.cz/research/fact-checking 
- Data from twitter_pl - https://zenodo.org/record/4319813 
- Data from getting_real_about_fake_news - https://www.kaggle.com/datasets/mrisdal/fake-news
- Data from oko.press - data, courtesy of OKO.press team (https://oko.press/) 
- Data from MagaCOV - https://github.com/UBC-NLP/megacov 

### Creation of weak supervised dataset
Idea behind this approach was inspired by paper [Weakly Supervised Learning for Fake News Detection on Twitter](https://ieeexplore.ieee.org/document/8508520) where authors created training dataset by assuming that all posts from trustworthy page is truth and from troll website is fake. As the extension of this method could be used some metrics from [Computational Propaganda in  Poland:  False Amplifiers and the Digital Public  Sphere](https://blogs.oii.ox.ac.uk/politicalbots/wp-content/uploads/sites/89/2017/06/Comprop-Poland.pdf), where various methods have been used to identify troll accounts.

## Methods

