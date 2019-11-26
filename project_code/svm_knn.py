from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# svm_classifier fun takes
  # writers_features -> 2D array (n_train_samples,features)
  # labels           -> 1D array (n_samples)
  # predict_features -> 2D array (n_test_samples,features)
# svm_classifier fun o/p
  #label classification for test writer
def svm_classifier(writers_features,labels,predict_features):
    #svm_clf = svm.SVC(gamma=0.001, C=1000,kernel='linear',degree=3)    #gamma is the learning rate , C is the tolerence
    svm_clf=LinearSVC(random_state=0,tol=1e-5,dual=False)
    #svm_clf=svm.SVC(kernel="rbf",gamma='auto',tol=1e-5,dual=False)
    X,Y = writers_features,labels
    svm_clf.fit(X,Y)
    writer_prediction=svm_clf.predict(predict_features)
    return writer_prediction


# knn_classifier fun takes
# writers_features -> 2D array (n_train_samples,features)
# labels           -> 1D array (n_samples)
# predict_features -> 2D array (n_test_samples,features)
# writers_features -> 2D array (n_train_samples,features)
  # labels           -> 1D array (n_samples)
  # predict_features -> 2D array (n_test_samples,features)
# knn_classifier fun o/p
  #label classification for test writer

def knn_classifier(writers_features,labels,predict_features):
    knn_clf = KNeighborsClassifier(n_neighbors=1)
    X, Y = writers_features, labels
    knn_clf.fit(X, Y)
    writer_prediction=knn_clf.predict(predict_features)
    return writer_prediction
def myknn(writers_features,labels,predict_features):
    min= np.linalg.norm(np.asarray(predict_features) - np.asarray(writers_features[0]))
    label=labels[0]
    for i in range(1,len(writers_features)):
        temp=np.linalg.norm(np.asarray(predict_features) - np.asarray(writers_features[i]))
        if(temp<min):
            min=temp
            label=labels[i]
    return label



