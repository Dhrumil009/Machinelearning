import numpy as np
import csv
from sklearn import neighbors
from numpy.random import shuffle

label_gen = {}
label_gen['male'] = 0
label_gen['female'] = 1

label_embarked = {}
label_embarked['C'] = 0
label_embarked['Q'] = 1
label_embarked['S'] = 2
label_embarked[''] = 0

def  load_data():

    traindata_features = []
    traindata_labels = []

    inputfile = open('data.csv')

    for line in inputfile.readlines():
        line = line.strip()
        if line == '':
            continue

        values = line.split(',')
        features = [values[i] for i in range(len(values) - 1)]

        features[0] = float(features[0])
        features[1] = float(label_gen[features[1]])
        if(features[2] == ""):
            features[2] = 0
        else:
            features[2] = float(features[2])

        features[3] = float(features[3])
        features[4] = float(features[4])

        if (features[5] == ""):
            features[5] = 0
        else:
            features[5] = float(features[5])

        features[6] = float(label_embarked[features[6]])

        traindata_features.append(features)
        traindata_labels.append(values[-1])

    traindata_features = np.array(traindata_features)
    traindata_labels = np.array(traindata_labels)

    return traindata_features, traindata_labels

def load_test_data():

    testdata_features = []

    inputfile = open('test.csv')

    for line in inputfile.readlines():
        line = line.strip()
        if line == '':
            continue

        values = line.split(',')

        # print(values1)
        features = [values[i] for i in range(len(values))]

        features[0] = float(features[0])
        features[1] = float(label_gen[features[1]])
        if (features[2] == ""):
            features[2] = 0
        else:
            features[2] = float(features[2])

        features[3] = float(features[3])
        features[4] = float(features[4])

        if (features[5] == ""):
            features[5] = 0
        else:
            features[5] = float(features[5])

        features[6] = float(label_embarked[features[6]])

        testdata_features.append(features)

    testdata_features = np.array(testdata_features)

    return  testdata_features

def do_nfold_cross_validation(traindata_features,traindata_labels,nfolds):
        X = traindata_features
        numobs = X.shape[0]

        all_ks = range(1,51)

        k_best = np.zeros(len(all_ks))
        num_ks = len(all_ks)

        all_errors = np.zeros([nfolds,num_ks])

        y = traindata_labels


        #1. first shuffle the data
        inds = list(range(numobs))
        shuffle(inds)
        X = X[inds,:]
        y = y[inds]


        #2. compute the split indeces
        split_inds = np.linspace(0,numobs-1,nfolds+1,dtype='int')


        #3. loop through all the ks
        for k_ind in range(num_ks):
            k = all_ks[k_ind]
            #print('Trying on k = %d' % k)
            #4. for each k do a whole CV experiment each fold experiment
            for exp_count in range(nfolds):
                #print('Doing sub-experiment for fold #%d' % (exp_count))

                #DEFINE TRAIN AND VALIDATION SETS
                part1 = list(range(split_inds[0],split_inds[exp_count]))
                part2 = list(range(split_inds[exp_count+1],split_inds[-1]))

                TRAINX = X[part1+part2,:]
                TRAINy = y[part1+part2]

                VALIDX = X[split_inds[exp_count]:split_inds[exp_count+1],:]
                VALIDy = y[split_inds[exp_count]:split_inds[exp_count+1]]


                #TRAIN KNN
                knn_classifier = neighbors.KNeighborsClassifier(k)
                knn_classifier.fit(TRAINX,TRAINy)


                #EVALUATE CLASSIFIER ON VALIDATION SET
                error = 0
                estimated_class = knn_classifier.predict(VALIDX)
                for i in range(len(estimated_class)):
                    if estimated_class[i] != VALIDy[i]:
                        error+=1

                error_rate = error*1.0/len(estimated_class)

                all_errors[exp_count,k_ind] = error_rate

        # print(all_errors)

        optimalk = choose_best_k(all_errors, all_ks)
        return optimalk


def choose_best_k(all_errors,all_ks):
        #choose the k that has the lowest average error times standard deviation
        #this gives us a combined measure of optimality: achieving low error while at the same time achieving low variance
        allmeans = np.mean(all_errors,0)
        allstds = np.std(all_errors,0)
        allmeanstimesstds = allmeans*allstds

        index_of_optimal_k = np.argmin(allmeanstimesstds)
        return all_ks[index_of_optimal_k]

def write_file(estimated_survival):
        print(len(estimated_survival))
        i = 0
        pasg_idList=[]
        output = open('output.csv')
        for line in output.readlines():
            line = line.strip()
            if line == '':
                continue

            values = line.split(',')
            pasg_idList.append(values[0])

        with open('output.csv', 'w', newline='') as testfile:
            writer = csv.writer(testfile)
            writer.writerow(['PassengerId', 'Survived'])
            for psg in pasg_idList:
                writer.writerow([psg, estimated_survival[i]])
                i=i+1

def main():

        traindata_features,traindata_labels = load_data()
        best_k=do_nfold_cross_validation(traindata_features,traindata_labels,10)
        print(best_k)

        testdata_features=load_test_data()
        knn_classifier = neighbors.KNeighborsClassifier(best_k)
        knn_classifier.fit(traindata_features, traindata_labels)

        estimated_survival = knn_classifier.predict(testdata_features)

        write_file(estimated_survival)





if __name__ == '__main__':
	main()