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

def fullPlot(mean, quantiles, name="Plot.png", CCC=0, num = 0):
    for i in range(1, 10):
        val = np.arange(4, 300.04, 0.04)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        #plt.fill_between(val, quantiles[0].flatten(), quantiles[1].flatten(),
        # print(val.shape)
        # print(mean[(i-1)*7401:i*7401].shape)
        plt.fill_between(val, quantiles[0][(i-1)*7401:i*7401], quantiles[1][(i-1)*7401:i*7401],
                        facecolor="powderblue", # The fill color
                        color='royalblue',       # The outline color
                        alpha=0.2)          # Transparency of the fill
        plt.plot(val, mean[(i-1)*7401:i*7401])
        ratings = getRatings('ratings_individual/arousal/dev_' + str(i) + '.csv')  
        for t in range(6):
            plt.plot(val, list(map(float, ratings[:, t])))
            # plt.plot(val, list(map(float, getRating(2))))
            # plt.plot(val, list(map(float, getRating(3))))
            # plt.plot(val, list(map(float, getRating(4))))
            # plt.plot(val, list(map(float, getRating(5))))
            # plt.plot(val, list(map(float, getRating(6))))
        plt.title("GPR Warped 6 Annotators F" + str(i))
        fig = plt.gcf()
        fig.set_size_inches((25, 10), forward=False)
        fig.savefig('WarpedFinal/' + name + '_Dev_' + str(i) + '.png', dpi=500)
        plt.close()
        #plt.show()
    f = open('WarpedFinal/Warped' + str(num) + '.txt', mode='w')
    f.write("Final CCC is " + CCC)
    f.write("\nValues:")
    f.write('\n' + str(m))
    f.write("\nLength Scales:")
    f.write('\n' + str(m.rbf.lengthscale))  
    f.write("\nInducing Inputs:")  
    f.write('\n' + str(m.inducing_inputs))
    f.close()

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

def MSE(predicted, actual, variance):
    n = len(predicted)
    minVar = min(variance)
    maxVar = max(variance)
    decile = (maxVar - minVar)/10
    print(n)
    print(maxVar)
    print(minVar)
    print(decile)
    MSEs = np.zeros(10, dtype=float)
    for i in range(0, len(variance)):
        print(variance[i]-minVar)
        print(np.floor((variance[i]-minVar)/decile))
        MSEs[int(np.floor((variance[i]-minVar)/decile))] += (actual - predicted)**2
    return MSEs/n
        
# extractFeatures('train_1.wav', 'features/PCA/train_1.csv', 1)
# extractFeatures('train_2.wav', 'features/PCA/train_2.csv', 1)
# extractFeatures('train_3.wav', 'features/PCA/train_3.csv', 1)
# extractFeatures('train_4.wav', 'features/PCA/train_4.csv', 1)
# extractFeatures('train_5.wav', 'features/PCA/train_5.csv', 1)
# extractFeatures('train_6.wav', 'features/PCA/train_6.csv', 1)
# exit()

#testFeat, testRatings = getInputs(file = "dev_")

# print(testRatings.shape)
kernel = GPy.kern.RBF(88, 1, 100, ARD=True)
X = np.genfromtxt('features/Full Feature/features_train.csv', delimiter=',')
Y = np.genfromtxt('features/Full Feature/ratings_train.csv', delimiter=',')
#X, Y = getInputs(file = "test_")
# print(X.shape)
# print(Y.shape)
m = loadFile(X, Y, 'WarpedFinal/Warped106.npy')
m_load = GPy.models.SparseGPRegression(X, Y, GPy.kern.RBF(88, 1, 100, ARD=True), infer=NewInference(), initialize = False)
m_load.update_model(False)
m_load.initialize_parameter()
m_load[:] = np.load('BasicFinal/Basic106.npy')
m_load.update_model(True)
#m = WarpedModelSimple(X, Y, kernel)
#m = GPy.models.SparseGPRegression(X, Y, kernel, infer=NewInference())

testFeat = np.genfromtxt('features/Full Feature/features_dev.csv', delimiter=',')
testRatingss = np.genfromtxt('features/Full Feature/ratings_dev.csv', delimiter=',')

testRatings = np.mean(testRatingss.astype(np.float64), axis = 1)

mean, variance, new = m.predict(testFeat)  
quantiles = m.predict_quantiles(testFeat)
# print(mean.flatten().shape)
# print(testRatings.flatten().shape)
exit()
print("Warped")
CCC = str(calculateCCC(mean.flatten().astype(np.float), testRatings.flatten().astype(np.float)))
print("CCC is " + CCC)
MSEval = MSE(mean, testRatings, variance)
print("MSE is " + MSEval)
SLPval = SLP(mean, variance, testRatingss)
print("SLP is " + str(SLPval))

mean, variance, new = m_load.predict(testFeat)  
quantiles = m_load.predict_quantiles(testFeat)
# print(mean.flatten().shape)
# print(testRatings.flatten().shape)
print("Basic:")
CCC = str(calculateCCC(mean.flatten().astype(np.float), testRatings.flatten().astype(np.float)))
print("CCC is " + CCC)
MSEval = MSE(mean, testRatings, variance)
print("MSE is " + str(MSEval))
SLPval = SLP(mean, variance, testRatingss)
print("SLP is " + str(SLPval))

exit()
#fullPlot(mean, quantiles, name=('Warped' + str(22)), CCC=CCC, num = 0)
x = 64
while (m.optimize(optimizer = 'SCG', messages=True, max_iters=200).status == 'maxiter exceeded'):
    mean, variance, new = m.predict(testFeat)  
    quantiles = m.predict_quantiles(testFeat)
    CCC = str(calculateCCC(mean.flatten().astype(np.float), testRatings.flatten().astype(np.float)))
    print("CCC is " + CCC)
    #fullPlot(mean, quantiles, name=('WarpedRecent'), CCC=CCC, num = x)
    f = open('BasicFinal/Basic' + str(x) + '.txt', mode='w')
    f.write("Final CCC is " + CCC)
    f.write("\nValues:")
    f.write('\n' + str(m))
    f.write("\nLength Scales:")
    f.write('\n' + str(m.rbf.lengthscale))  
    f.write("\nInducing Inputs:")  
    f.write('\n' + str(m.inducing_inputs))
    f.close()
    saveModel(m, name=('WarpedFinal/Warped' + str(x) + '.npy'))
    x += 1
    

print('Finished')
print('------------------------------')
mean, variance, new = m.predict(testFeat)  
quantiles = m.predict_quantiles(testFeat)
CCC = str(calculateCCC(mean.flatten().astype(np.float), testRatings.flatten().astype(np.float)))
print("Final CCC is " + CCC)
saveModel(m, name=('WarpedFinal/Final.npy'))
fullPlot(mean, quantiles, name=('Warped' + str(x)), num = x, CCC=CCC)

