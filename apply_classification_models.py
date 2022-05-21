# -*- coding: utf-8 -*-
# Code by: Faisal Ashraf 
# 32 classification models 

# call the function make_classification(df, cls_label, tst_sz = 0.25) 

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier  
from sklearn.linear_model import Perceptron

from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB  
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

names = [
         "LogisticRegression",
         "LogisticRegressionCV",
         "SGDClassifier",
         "RidgeClassifier",
         "RidgeClassifierCV",
         "PassiveAggressiveClassifier",
         "Perceptron",

         "SVC",
         "NuSVC",
         "LinearSVC",

         "GaussianNB",
         "MultinomialNB",
         "BernoulliNB",
         "CategoricalNB",
         "ComplementNB",

         "DecisionTreeClassifier",
         "ExtraTreeClassifier",
         
         "VotingClassifier",
         "AdaBoostClassifier",
         "GradientBoostingClassifier",
         "BaggingClassifier",
         "ExtraTreesClassifier",
         "RandomForestClassifier",
         "StackingClassifier",

         "NearestCentroid",
         "KNeighborsClassifier",

         "CalibratedClassifierCV",
         "LinearDiscriminantAnalysis",
         "QuadraticDiscriminantAnalysis",

         "LabelPropagation",
         "LabelSpreading",

         "MLPClassifier"
]



classifiers = [
               LogisticRegression(),
               LogisticRegressionCV(cv=5),
               make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)),
               RidgeClassifier(),
               RidgeClassifierCV(),
               PassiveAggressiveClassifier(),
               Perceptron(), 

               make_pipeline(StandardScaler(), SVC(gamma='auto')),
               make_pipeline(StandardScaler(), NuSVC()),
               make_pipeline(StandardScaler(), LinearSVC()),

               GaussianNB(),
               MultinomialNB(),
               BernoulliNB(),
               CategoricalNB(),
               ComplementNB(),

               DecisionTreeClassifier(),
               ExtraTreeClassifier(),

               VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('gnb', GaussianNB())], voting='hard'),
               AdaBoostClassifier(n_estimators=100),
               GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1),
               BaggingClassifier(base_estimator=SVC(), n_estimators=10 ),
               ExtraTreesClassifier(n_estimators=100),
               RandomForestClassifier(),
               StackingClassifier(estimators= [ ('rf', RandomForestClassifier(n_estimators=100)),('svr', make_pipeline(StandardScaler(), LinearSVC()))], final_estimator=LogisticRegression()),


               NearestCentroid(),
               KNeighborsClassifier(n_neighbors=3),


               CalibratedClassifierCV(base_estimator=GaussianNB(), cv=5),
               LinearDiscriminantAnalysis(),
               QuadraticDiscriminantAnalysis(),

               LabelPropagation(),
               LabelSpreading(), 

               MLPClassifier()
            ]

# iterate over classifiers
def make_classification(df, cls_label, tst_sz = 0.25):
  from sklearn import metrics
  results = {'model':[], 'accuracy':[], 'balanced_accuracy': [], 'precision':[], 'recall': [], 'f1score':[]  }
  
  y = df[cls_label]
  X = df.drop(columns = [cls_label] )
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= tst_sz, random_state = 2)
  
  for model, clf in zip(names, classifiers):
  #   ax = plt.subplot(1, len(classifiers) + 1, i)
      print("Applying "+ model)
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      accuracy = metrics.accuracy_score(y_test, y_pred)
      balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
      precision = metrics.precision_score(y_test, y_pred, average='weighted')
      recall = metrics.recall_score(y_test, y_pred, average='weighted')
      f1score = metrics.f1_score(y_test, y_pred, average='weighted')
      
      results['model'].append(model)
      results['accuracy'].append(accuracy)
      results['balanced_accuracy'].append(balanced_accuracy)
      results['precision'].append(precision)
      results['recall'].append(recall)
      results['f1score'].append(f1score)

      import pandas as pd 
      scores = pd.DataFrame(results)
  return scores

