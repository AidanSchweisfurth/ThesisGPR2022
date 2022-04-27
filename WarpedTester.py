from scipy.io import wavfile
from sklearn.decomposition import PCA
import math
import pandas as pd
import GPy
import csv
import numpy as np
import matplotlib.pyplot as plt
from warped_model_Annotators import WarpedModelSimple
import opensmile
from inference import NewInference
import scipy

from warpingFunction import WarpingFunction
testRatingsFile = 'ratings_individual/arousal/train_2.csv'

def extractRating(file):
    vals = np.empty(shape=[0, 6])
    f = open('ratings_individual/arousal/' + file, 'rt')
    data = csv.reader(f)
    for row in data:
        x = row[0].split(';')
        val = x[1:]
        vals = np.vstack((vals, [val]))
    my_data = np.genfromtxt('features/Full Feature/' + file, delimiter=',')
    f.close()
    return my_data, vals[101:]

def getInputs(file='train_'):
    features, ratings = extractRating(file + '1.csv')
    feat, rate = extractRating(file + '2.csv')
    features = np.vstack((features, feat))
    ratings = np.vstack((ratings, rate))
    feat, rate = extractRating(file + '3.csv')
    features = np.vstack((features, feat))
    ratings = np.vstack((ratings, rate))
    feat, rate = extractRating(file + '4.csv')
    features = np.vstack((features, feat))
    ratings = np.vstack((ratings, rate))
    feat, rate = extractRating(file + '5.csv')
    features = np.vstack((features, feat))
    ratings = np.vstack((ratings, rate))
    feat, rate = extractRating(file + '6.csv')
    features = np.vstack((features, feat))
    ratings = np.vstack((ratings, rate))
    feat, rate = extractRating(file + '7.csv')
    features = np.vstack((features, feat))
    ratings = np.vstack((ratings, rate))
    feat, rate = extractRating(file + '8.csv')
    features = np.vstack((features, feat))
    ratings = np.vstack((ratings, rate))
    feat, rate = extractRating(file + '9.csv')
    features = np.vstack((features, feat))
    ratings = np.vstack((ratings, rate))
    np.savetxt('features/Full Feature/features_dev.csv', features, delimiter=",")
    np.savetxt('features/Full Feature/ratings_dev.csv', ratings.astype(float), delimiter=",")
    return features, ratings

def getRaw():
    ratings = np.empty(shape=[0, 6])
    f = open(testRatingsFile, 'rt')
    data = csv.reader(f)
    for row in data:
        x = row[0].split(';')
        val = x[1:]
        ratings = np.vstack((ratings, [val]))
    vals = ratings[101:]
    f.close()
    return vals

def getRatings(testRatingsFile):
    ratings = np.empty(shape=[0, 6])
    f = open(testRatingsFile, 'rt')
    data = csv.reader(f)
    for row in data:
        x = row[0].split(';')
        val = x[1:]
        ratings = np.vstack((ratings, [val]))
    vals = ratings[101:]
    f.close()
    return vals

def getRating(rater=1, testRatingsFile=''):
    ratings = np.empty(shape=[0, 1])
    f = open(testRatingsFile, 'rt')
    data = csv.reader(f)
    for row in data:
        x = row[0].split(';')
        val = x[rater]
        ratings = np.vstack((ratings, [val]))
    vals = ratings[101:]
    f.close()
    return vals

def calculateCCC(predict, standard):
    cor=np.corrcoef(standard,predict)[0][1]

    meanStand=np.mean(standard)
    meanPred=np.mean(predict)
    
    varStand=np.var(standard)
    varPred=np.var(predict)
    
    sdStand=np.std(standard)
    sdPred=np.std(predict)
    
    numerator=2*cor*sdStand*sdPred
    denominator=varStand+varPred+(meanStand-meanPred)**2

    return numerator/denominator

def SLP(mean, variance, annotators):
    total = 0
    for i in range(0,5):
        for t in range(len(mean)):
            val = scipy.stats.lognorm(mean[t], variance[t]).pdf(annotators[t][i])
            if math.isnan(val):
                val = 0
            total += val

    return total

def fullPlot(mean, quantiles, name="Plot.png", CCC=0, num = 0, mean2=None, quantiles2=None):
    for i in range(1, 10):
        val = np.arange(4, 300.04, 0.04)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        #plt.fill_between(val, quantiles[0].flatten()[(i-1)*7401:i*7401], quantiles[1].flatten()[(i-1)*7401:i*7401],
        plt.fill_between(val, quantiles[0][(i-1)*7401:i*7401], quantiles[1][(i-1)*7401:i*7401],
                        facecolor="powderblue", # The fill color
                        color='royalblue',       # The outline color
                        alpha=0.2, label='95%% Confidence Interval')          # Transparency of the fill
        #plt.plot(val, quantiles[0][(i-1)*7401:i*7401])
        #plt.plot(val, quantiles[1][(i-1)*7401:i*7401])
        # plt.plot(val, mean2[(i-1)*7401:i*7401], color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))
        plt.plot(val, mean[(i-1)*7401:i*7401], label='Mean Prediction')
        # plt.plot(val, quantiles2[0].flatten()[(i-1)*7401:i*7401], color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765))
        # plt.plot(val, quantiles[0][(i-1)*7401:i*7401], color='Orange', label='Warped')
        # plt.plot(val, quantiles2[1].flatten()[(i-1)*7401:i*7401], color=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), label='Basic')
        # plt.plot(val, quantiles[1][(i-1)*7401:i*7401], color='Orange')
        # #plt.fill_between(val, quantiles2[0].flatten()[(i-1)*7401:i*7401], quantiles[1].flatten()[(i-1)*7401:i*7401],
        # plt.fill_between(val, quantiles[0][(i-1)*7401:i*7401], quantiles[1][(i-1)*7401:i*7401],
        #                 facecolor="lightcoral", # The fill color
        #                 color='firebrick',       # The outline color
        #                 alpha=0.2)          # Transparency of the fill
        #plt.plot(val, quantiles2[0][(i-1)*7401:i*7401])
        #plt.plot(val, quantiles2[1][(i-1)*7401:i*7401])
        #plt.plot(val, mean2[(i-1)*7401:i*7401])
        ratings = getRatings('ratings_individual/arousal/dev_' + str(i) + '.csv')  
        for t in range(6):
            plt.plot(val, list(map(float, ratings[:, t])))
            # plt.plot(val, list(map(float, getRating(2))))
            # plt.plot(val, list(map(float, getRating(3))))
            # plt.plot(val, list(map(float, getRating(4))))
            # plt.plot(val, list(map(float, getRating(5))))
            # plt.plot(val, list(map(float, getRating(6))))
        plt.title("Warped Dev " + str(i) + " Regression", fontsize=24)
        plt.xlabel('Time(Seconds)', fontsize=20)
        plt.ylabel("Arousal", fontsize=20)
        plt.legend(fontsize=20)
        fig = plt.gcf()
        fig.set_size_inches((25, 10), forward=False)
        fig.savefig('Tests/' + name + '_Dev_' + str(i) + '.png', dpi=500)
        #plt.close()
        plt.show()

def extractFeatures(filename, output, id):
    fs, data = wavfile.read(filename)
    window = 4
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    vals = smile.process_file(filename, start=0, end=window)
    array = pd.DataFrame(vals).to_numpy()
    #vals.to_csv(output, index=False, mode = 'x', index_label = True)
    print("Starting...")
    print("-------------------------")
    for x in range(int(4.04*100), math.ceil((len(data)/fs))*100 + 1, int(0.04*100)):
        i = x/100
        vals = smile.process_file(filename, start=i-window, end=i)
        array = np.concatenate((array, pd.DataFrame(vals).to_numpy()), axis=0)
        #vals.to_csv(output, index=False, mode = 'a', index_label = False, header = False)
        if (i == 40 or i == 80 or i == 120 or i == 160 or i == 200 or i == 240 or i==280):
            print("Progress " + str(i))

    pca = PCA(n_components=40)
    pca.fit(array)
    new = pca.transform(array)
    np.savetxt(output, new, delimiter=",")
    print("Completed")

def loadFile(X, Y, name='tester.npy'):
    m_load = WarpedModelSimple(X, Y, GPy.kern.RBF(88, 1, 100, ARD=True), initialize = False)
    m_load.update_model(False)
    m_load.initialize_parameter()
    m_load[:] = np.load(name)
    m_load.update_model(True)
    return m_load

def saveModel(m, name='tester.npy'):
    np.save(name, m.param_array)

def MSE(predicted, predicted2, actual, Y, name=""):
    n = np.ceil(len(predicted)/10)
    variance = Y
    minVar = variance.min()
    maxVar = variance.max()
    decile = (maxVar - minVar)/10
    MSEs = np.zeros(variance.size, dtype=float)
    for i in range(0, variance.size):
        MSEs[i] = (actual[i] - predicted[i])**2
    
    final = np.sort(MSEs)
    final = np.array_split(final, 10)
    i = 0
    for a in final:
        final[i] = a.sum()
        i += 1

    totals = np.zeros(10, dtype=float)
    i = 1

    for p in range(len(totals)):
        totals[p] += np.sum(final[:i])
        i += 1

    i = 1
    for p in range(len(totals)):
        totals[p] = totals[p] / (n*i)
        i += 1

    final = final/n
#######################################################
    MSEs2 = np.zeros(variance.size, dtype=float)
    for i in range(0, variance.size):
        MSEs2[i] = (actual[i] - predicted2[i])**2
    
    final2 = np.sort(MSEs2)
    final2 = np.array_split(final2, 10)
    i = 0
    for a in final2:
        final2[i] = a.sum()
        i += 1

    totals2 = np.zeros(10, dtype=float)
    i = 1

    for p in range(len(totals2)):
        totals2[p] += np.sum(final2[:i])
        i += 1
    
    i = 1
    for p in range(len(totals2)):
        totals2[p] = totals2[p] / (n*i)
        i += 1

    final2 = final2/n
    #############################################

    axis = np.arange(minVar, maxVar, decile)
    labels = []
    for a in range(len(axis)):
        labels.append(str(axis[a]-decile/2) + '-' + str(axis[a] + decile/2))
    width = (np.max(axis) - np.min(axis))/30
    plt.plot(axis,final, linestyle='-', marker='x')
    plt.bar(axis, totals, width=width)
    plt.title('Warped MSE Plot')
    plt.xlabel('Variances')
    plt.xticks(axis)
    plt.ylabel('Error')
    plt.ylim(0, 0.17)
    fig = plt.gcf()
    fig.set_size_inches((18, 10), forward=False)
    fig.savefig('End Results/T-' + name + 'MSEWarped.png', dpi=500)
    plt.close()

    plt.plot(axis,final2, linestyle='-', marker='x')
    plt.bar(axis, totals2, width=width)
    plt.title('Basic MSE Plot')
    plt.xlabel('Variances')
    plt.xticks(axis)
    plt.ylabel('Error')
    plt.ylim(0, 0.17)
    fig = plt.gcf()
    fig.set_size_inches((18, 10), forward=False)
    fig.savefig('End Results/T-' + name + 'MSEBasic.png', dpi=500)
    plt.close()
    
    
    plt.plot(axis,final2, linestyle='-', marker='o')
    plt.plot(axis,final, linestyle='-', marker='x')
    plt.bar([i-0.5*width for i in axis], totals2, width=width, label='Basic')
    plt.bar([i+0.5*width for i in axis], totals, width=width, label='Warped')
    plt.xticks(axis)
    plt.ylim(0, 0.17)
    #plt.xlim(0.01, 0.41)
    plt.title('MSE Plot', fontsize=20)
    plt.xlabel('Standard Deviation', fontsize=18)
    plt.ylabel('Error', fontsize=18)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches((18, 10), forward=False)
    fig.savefig('End Results/R-' + name + 'MSE.png', dpi=500)
    plt.close()
    print("Warped MSE is " + str(final))
    print("Basic MSE is " + str(final2))

        

kernel = GPy.kern.RBF(88, 1, 100, ARD=True)
X = np.genfromtxt('features/Full Feature/features_train.csv', delimiter=',')
Y = np.genfromtxt('features/Full Feature/ratings_train.csv', delimiter=',')

#X, Y = getInputs(file = "test_")
# print(X.shape)
# print(Y.shape)
mW = loadFile(X, Y, 'WarpedFinal/Final.npy')
mB = GPy.models.SparseGPRegression(X, Y, GPy.kern.RBF(88, 1, 100, ARD=True), infer=NewInference(), initialize = False)
mB.update_model(False)
mB.initialize_parameter()
mB[:] = np.load('BasicFinal/Final.npy')
mB.update_model(True)

testFeat = np.genfromtxt('features/Full Feature/features_dev.csv', delimiter=',')
testRatingss = np.genfromtxt('features/Full Feature/ratings_dev.csv', delimiter=',')

testRatings = np.mean(testRatingss.astype(np.float64), axis = 1)

mean1, variance, new = mW.predict(testFeat) 
quantiles = mW.predict_quantiles(testFeat) 
mean2, variance = mB.predict(testFeat)  
quantiles2 = mB.predict_quantiles(testFeat) 
fullPlot(mean1, quantiles, name=('Warped'), quantiles2=quantiles2, mean2=mean2)
#MSE(mean1, mean2, testRatings, np.std(testRatingss, axis=1))
exit()
#### Setup ####
values =  np.column_stack((np.std(testRatingss, axis=1), testRatings.flatten().astype(np.float)))
values = np.column_stack((values, mean1.flatten().astype(np.float)))
values = np.column_stack((values, mean2.flatten().astype(np.float)))
values = np.column_stack((values, np.abs(testRatings)))

array = np.zeros((1, 2))
for a in testRatingss:
    array = np.vstack((array, scipy.stats.t.interval(alpha=0.95, df=len(a)-1, loc=np.mean(a), scale=scipy.stats.sem(a))))
array = array[1:, :]
values = np.column_stack((values, np.maximum(np.abs(array[:, 0]), np.abs(array[:, 1]))))

#### Biggest 10% of std of annotations ####
values = values[values[:, 0].argsort()]
CCC = str(calculateCCC(values[59408:, 2], values[59408:, 1]))
print("CCC Warped 10% std is " + CCC)
CCC = str(calculateCCC(values[59408:, 3], values[59408:, 1]))
print("CCC Basic 10% std is " + CCC)
MSE(values[59408:, 2], values[54908:, 3], values[59408:, 1], values[59408:, 0], name="STD")

#### Biggest 10% of mean of annotations ####
values = values[values[:, 4].argsort()]
CCC = str(calculateCCC(values[59408:, 2], values[59408:, 1]))
print("CCC Warped 10% mean is " + CCC)
CCC = str(calculateCCC(values[59408:, 3], values[59408:, 1]))
print("CCC Basic 10% mean is " + CCC)
MSE(values[59408:, 2], values[59408:, 3], values[59408:, 1], values[59408:, 0], name="Mean")

#### Ones where percentiles are highest ####
values = values[values[:, 5].argsort()]
CCC = str(calculateCCC(values[59408:, 2], values[59408:, 1]))
print("CCC Warped highest Quantiles is " + CCC)
CCC = str(calculateCCC(values[59408:, 3], values[59408:, 1]))
print("CCC Basic highest Quantiles is " + CCC)
MSE(values[59408:, 2], values[59408:, 3], values[59408:, 1], values[59408:, 0], name="quantiles")

