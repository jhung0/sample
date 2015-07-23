############################
#Model for taking reviews and predicting sentiments (negative or positive)
#Uses labeled and unlabeled data to learn distributed representations of words (Word2Vec algorithm), training done with Random forest classifier
#Evaluated with ROC curves
############################
import pip
def install(package):
   pip.main(['install', package])
install('gensim')
install('cython')
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk.data
nltk.download()  
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn import metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import roc_curve, auc
import pylab as pl

#Returns a list of words from a review document, string of text, can remove stop words
def createWordList(review, remove_stopwords=False ):
    reviewText = BeautifulSoup(review).get_text()#Remove html, nonletters
    reviewText = re.sub('[^a-zA-Z]',' ', reviewText)#reviewText.replace('[^a-zA-Z]',' ')

    words = reviewText.lower().split()
    if remove_stopwords:
        words = [w for w in words if not w in set(stopwords.words('english'))]
    return(words)


#Returns a list of lists (sentences split into words) from a review document, string of text
def createSentenceList(review, remove_stopwords=False ):
    review = BeautifulSoup(review).get_text()
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(review.strip())

    sentencesList = []
    for sentence in sentences:
        if len(sentence) > 0:
            sentencesList.append(createWordList(sentence, remove_stopwords))
    return sentencesList
  
# Given a set of reviews, return the average feature vector for each one  
def createAvgFeatureVector(reviews, model, num_features):
    
    reviewFeatureVecs = np.zeros((len(reviews),num_features))
    
    review_wordlist = []#list of review wordlists
    for review in reviews:
        review_wordlist.append(createWordList( review, True ))
        
    for count,review in enumerate(review_wordlist):
        featureVec = np.zeros((num_features,1))
        index2word_set = set(model.index2word)#set of words in model vocab

        model_review_words = [model[word] for word in review if word in index2word_set]# If a word is in the model's vocab, add its feature vector to the total
        nwords = len(model_review_words)
        featureVec = sum(np.array(model_review_words))
        reviewFeatureVecs[count] = np.divide(featureVec,nwords)

    return reviewFeatureVecs

#### Import unlabeled, labeled data
labeled = pd.read_csv('labeledTrainData.tsv', header=0,delimiter='\t', quoting=3)
unlabeled = pd.read_csv('unlabeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

# Set values for parameters for word2vec
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Any word that does not occur at least this many times across all documents is ignored                       
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
    
####Do 3 fold cross validation: Split labeled data into training and test data, change reviews into word vectors
kf=cross_validation.KFold(len(labeled['review']), n_folds=3, indices=None, shuffle=True, random_state=None)
for fold,(train_index, test_index) in enumerate(kf):
    labeled_train, labeled_test = np.array(labeled)[train_index], np.array(labeled)[test_index]

    allSentenceList = []
    labeled_train_reviews = [x[2] for x in labeled_train]
    for review in labeled_train_reviews :
        allSentenceList += createSentenceList(review)
        
    unlabeled_reviews = [x[1] for x in unlabeled]
    for review in unlabeled_reviews:
        allSentenceList += createSentenceList(review)

    # Initialize and train the model for obtaining distributed word vectors 
    model = word2vec.Word2Vec(allSentenceList, workers=num_workers, size=num_features, min_count = min_word_count,  window = context, sample = downsampling)
    # Post training
    model.init_sims(replace=True)

    # reload it using Word2Vec.load()
    model_name = '300f_40m_10c'
    model.save(model_name)

    #Turn word vectors into feature set for training by averaging for each review
    train_data= createAvgFeatureVector(labeled_train_reviews, model, num_features )
    test_data= createAvgFeatureVector([x[2] for x in labeled_test], model, num_features )
    
    #Classify with random forest, 100 trees
    rf = RandomForestClassifier(n_estimators = 100) 
    rf = rf.fit(train_data , [x[1] for x in labeled_train])

    #Area under ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(np.array([x[1] for x in labeled_test]), np.array(rf.predict_proba(test_data ))[:,1])
    roc_auc = auc(fpr, tpr)
    print('Area under ROC curve: ', roc_auc)
    
    #Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='Word vector ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC Curves')
    pl.legend(loc="lower right")
    pl.show()

