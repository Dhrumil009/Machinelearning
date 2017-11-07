import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import shuffle
from sklearn import grid_search
from sklearn import cross_validation
from sklearn import datasets
from random import random
from sklearn import svm
import csv
from sklearn import neighbors


def load_data():
    traindata_features = []
    traindata_labels = []

    inputfile = open('mushrooms.csv')
    label_code = {}

    label_code['a'] = 0;    label_code['b'] = 1;    label_code['c'] = 2
    label_code['d'] = 3;    label_code['e'] = 4;    label_code['f'] = 5
    label_code['g'] = 6;    label_code['h'] = 7;    label_code['i'] = 8
    label_code['j'] = 9;    label_code['k'] = 10;    label_code['l'] = 11
    label_code['m'] = 12;    label_code['n'] = 13;    label_code['o'] = 14
    label_code['p'] = 15;    label_code['q'] = 16;    label_code['r'] = 17
    label_code['s'] = 18;    label_code['t'] = 19;    label_code['u'] = 20
    label_code['v'] = 21;    label_code['w'] = 22;    label_code['x'] = 23
    label_code['y'] = 24;    label_code['z'] = 25;  label_code[''] = 26
    label_code['?'] = 27


    for line in inputfile.readlines():
        line = line.strip()
        if line == '':
            continue

        values = line.split(',')
        features = [label_code[values[i]] for i in range(len(values) - 1)]

        traindata_features.append(features)
        traindata_labels.append(label_code[values[-1]])

    traindata_features = np.array(traindata_features)
    traindata_labels = np.array(traindata_labels)
    return traindata_features, traindata_labels




def do_nfold_cross_validation_KNN(traindata_features, traindata_labels, nfolds):
    X = traindata_features
    y = traindata_labels

    numobs = X.shape[0]

    train_portion = 0.3
    all_ks = range(1, 51)
    k_best = np.zeros(len(all_ks))
    num_of_expr = 100

    for exp_count in range(num_of_expr):

        print('Doing experiment #%d' % (exp_count))
        inds = list(range(numobs))
        shuffle(inds)
        X = X[inds, :]
        y = [y[i] for i in inds]

        TRAINX = X[:int(numobs * train_portion), :]
        TRAINy = y[:int(numobs * train_portion)]

        VALIDX = X[int(numobs * train_portion):, :]
        VALIDy = y[int(numobs * train_portion):]

        allerrors = []

        for k in all_ks:
            if k > len(TRAINy):
                break
            knn_classifier = neighbors.KNeighborsClassifier(k)
            knn_classifier.fit(TRAINX, TRAINy)

            error = 0
            estimated_class = knn_classifier.predict(VALIDX)
            for i in range(len(estimated_class)):
                if estimated_class[i] != VALIDy[i]:
                    error += 1

            error_rate = error * 1.0 / len(estimated_class)

            allerrors.append([k, error_rate])

        allerrors = np.array(allerrors)

        bestk = allerrors[allerrors[:, 1] == np.min(allerrors[:, 1]), 0]
        for k in bestk:
            k_best[all_ks.index(k)] += 1

    plt.bar(all_ks, k_best)
    plt.show()

def do_nfold_cross_validation_LogisticRegression(traindata_features, traindata_labels, nfolds):
    X = traindata_features
    numobs = X.shape[0]

    all_Cs = np.logspace(-8, 8, 1000, base=2)
    C_best = np.logspace(-8, 8, 1000, base=2)
    num_Cs = len(all_Cs)

    all_errors = np.zeros(num_Cs)

    y = np.array(traindata_labels)


    inds = list(range(numobs))
    shuffle(inds)
    X = X[inds, :]
    y = y[inds]

    for C_ind in range(num_Cs):
        C = all_Cs[C_ind]
        model = linear_model.LogisticRegression(C=C)

        loo_cv = cross_validation.LeaveOneOut(numobs)  # Leave-one out

        predictions = cross_validation.cross_val_predict(model, X, y, cv=loo_cv)
        error = 0
        for i in range(numobs):
            if predictions[i] != y[i]:
                error += 1

        error_rate = error * 1.0 / numobs

        all_errors[C_ind] = error_rate  # [1-v for v in scores]

        Co_best=all_errors[all_errors == np.min(all_errors)]
        for k in Co_best:
            C_best[range(num_Cs).index(k)] += 1

    plt.bar(all_Cs,all_errors)
    plt.show()

def summarize_CV_results(all_errors,all_Cs):
	allmeans = np.mean(all_errors,0)
	allstds = np.std(all_errors,0)
	allmeanstimesstds = allmeans*allstds
	allmedians = np.median(all_errors,0)
	alliqrs = np.percentile(all_errors,75,0)-np.percentile(all_errors,25,0)

	plt.plot(all_Cs,allmeans,'r-',label='means')
	plt.plot(all_Cs,allstds,'b-',label='stds')
	plt.plot(all_Cs,allmedians,'g-',label='medians')
	plt.plot(all_Cs,alliqrs,'m-',label='iqrs')
	plt.plot(all_Cs,allmeanstimesstds,'k-',label='allmeanstimesstds')
	plt.legend()
	plt.xticks(all_Cs)
	plt.show()

def use_grid_search_LogisticRegression(traindata_features, traindata_labels, nfolds):
    print("in")
    X = traindata_features
    numobs = X.shape[0]

    hyperparam_grid = [{'C': np.logspace(-12, 12, 10, base=2)}]
    y = np.array(traindata_labels)
    print("in 1")
    # 1. first shuffle the data
    inds = list(range(numobs))
    shuffle(inds)
    X = X[inds, :]
    y = y[inds]

    print("in 2")
    cv = cross_validation.LeaveOneOut(numobs)
    grid_searcher = grid_search.GridSearchCV(linear_model.LogisticRegression(), hyperparam_grid, cv=cv)
    grid_searcher.fit(X, y)
    print("in 3")
    print(grid_searcher.best_score_)
    return grid_searcher.best_estimator_

def use_grid_search_svm(traindata_features,traindata_labels,nfolds):
	X = traindata_features[:,0:2]
	numobs = X.shape[0]

	hyperparam_grid = [{'C':np.logspace(-12,12,100,base=2),'gamma':np.logspace(-12,12,100,base=2)}]
	y = np.array(traindata_labels)


	#1. first shuffle the data
	inds = list(range(numobs))
	shuffle(inds)
	X = X[inds,:]
	y = y[inds]

	cv = cross_validation.LeaveOneOut(numobs)
	grid_searcher = grid_search.GridSearchCV(svm.SVC(kernel='rbf'),hyperparam_grid,cv=cv)
	grid_searcher.fit(X,y)

	print(grid_searcher.best_score_)
	return grid_searcher.best_estimator_

def plot_CV_results(all_errors,all_Cs):
	plt.boxplot(all_errors,labels = all_Cs,showmeans=True)
	plt.show()

def main():
    traindata_features,traindata_labels=load_data()
    do_nfold_cross_validation_KNN(traindata_features, traindata_labels, 10)
    ptimalC = do_nfold_cross_validation_LogisticRegression(traindata_features, traindata_labels, 10)

if __name__ == '__main__':
    main()