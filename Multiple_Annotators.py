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
testRatingsFile = 'ratings_individual/arousal/dev_2.csv'

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
    # feat, rate = extractRating(file + '5.csv')
    # features = np.vstack((features, feat))
    # ratings = np.vstack((ratings, rate))
    # feat, rate = extractRating(file + '6.csv')
    # features = np.vstack((features, feat))
    # ratings = np.vstack((ratings, rate))
    # feat, rate = extractRating(file + '7.csv')
    # features = np.vstack((features, feat))
    # ratings = np.vstack((ratings, rate))
    # feat, rate = extractRating(file + '8.csv')
    # features = np.vstack((features, feat))
    # ratings = np.vstack((ratings, rate))
    # feat, rate = extractRating(file + '9.csv')
    # features = np.vstack((features, feat))
    # ratings = np.vstack((ratings, rate))
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

def getRatings():
    ratings = np.empty(shape=[0, 6])
    f = open(testRatingsFile, 'rt')
    data = csv.reader(f)
    for row in data:
        x = row[0].split(';')
        val = x[1:]
        ratings = np.vstack((ratings, [val]))
    vals = ratings[101:]
    f.close()
    return np.mean(vals.astype(np.float64), axis=1)

def getRating(rater=1):
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

def fullPlot(mean, quantiles, name="Plot.png", model=None, CCC=0, num = 0):
    val = np.arange(4, 300.04, 0.04)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #plt.fill_between(val, quantiles[0].flatten(), quantiles[1].flatten(),
    plt.fill_between(val, quantiles[0], quantiles[1],
                    facecolor="powderblue", # The fill color
                    color='royalblue',       # The outline color
                    alpha=0.2)          # Transparency of the fill
    plt.plot(val, mean)  
    plt.plot(val, list(map(float, getRating(rater=1))))
    plt.plot(val, list(map(float, getRating(2))))
    plt.plot(val, list(map(float, getRating(3))))
    plt.plot(val, list(map(float, getRating(4))))
    plt.plot(val, list(map(float, getRating(5))))
    plt.plot(val, list(map(float, getRating(6))))
    plt.title("GPR Warped 6 Annotators")
    fig = plt.gcf()
    fig.set_size_inches((20, 10), forward=False)
    fig.savefig('WarpedFinal//' + name, dpi=500)
    plt.close()
    #plt.show()
    f = open('WarpedFinal//Warped' + str(num) + '.txt', mode='w')
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

    # pca = PCA(n_components=40)
    # pca.fit(array)
    # new = pca.transform(array)
    np.savetxt(output, array, delimiter=",")
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
    MSEs = np.zeros(10, dtype=float)
    for i in range(0, len(variance)):
        MSEs[variance[i]//decile] += (actual - predicted)**2
    return MSEs/n
        
# extractFeatures('train_4.wav', 'features/Full Feature/train_4.csv', 1)
# extractFeatures('train_5.wav', 'features/Full Feature/train_5.csv', 1)
# extractFeatures('train_6.wav', 'features/Full Feature/train_6.csv', 1)
# extractFeatures('train_7.wav', 'features/Full Feature/train_7.csv', 1)
# extractFeatures('train_8.wav', 'features/Full Feature/train_8.csv', 1)
# extractFeatures('train_9.wav', 'features/Full Feature/train_9.csv', 1)
# exit()

testFeat = np.genfromtxt('features/Full Feature/dev_2.csv', delimiter=',')
rawRatings = getRaw()
testRatings = getRatings()
#print(testRatings.shape)
kernel = GPy.kern.RBF(88, 1, 100, ARD=True)
X, Y = getInputs()

# print(Y.shape)
# print(X.shape)

m = loadFile(X, Y, 'Warped1F/Final.npy')

#m = WarpedModelSimple(X, Y, kernel)

#m = GPy.models.WarpedGP(X, Y, kernel)
#m = GPy.models.SparseGPRegression(X, Y, kernel, infer=NewInference())
#mean, variance = m.predict(testFeat)  
#quantiles = m.predict_quantiles(testFeat)
#print(quantiles)
#print(testRatings)
#print(mean)
#print("CCC is " + str(calculateCCC(mean.flatten().astype(np.float), testRatings.astype(np.float))))

#m.optimize_restarts(num_restarts = 5, optimizer = 'SCG', max_iters=200)
mean, variance, new = m.predict(testFeat)  
quantiles = m.predict_quantiles(testFeat)
CCC = str(calculateCCC(mean.flatten().astype(np.float), testRatings.astype(np.float)))
print("CCC is " + CCC)
fullPlot(mean, quantiles, model=m, name=('Warped' + str(0) + '.png'), CCC=CCC, num = 0)
x = 1
while (m.optimize(optimizer = 'SCG', messages=True, max_iters=2000).status == 'maxiter exceeded'):
    mean, variance, new = m.predict(testFeat)  
    quantiles = m.predict_quantiles(testFeat)
    CCC = str(calculateCCC(mean.flatten().astype(np.float), testRatings.flatten().astype(np.float)))
    print("CCC is " + CCC)
    fullPlot(mean, quantiles, model=m, name=('Warped' + str(x) + '.png'), CCC=CCC, num = x)
    saveModel(m, name=('WarpedFinal//Warped' + str(x) + '.npy'))
    x += 1
    

print('Finished')
print('------------------------------')
mean, variance, new = m.predict(testFeat)  
quantiles = m.predict_quantiles(testFeat)
CCC = str(calculateCCC(mean.flatten().astype(np.float), testRatings.flatten().astype(np.float)))
print("Final CCC is " + CCC)
# sum = SLP(mean.astype(np.float), variance.astype(np.float), rawRatings.astype(np.float))
# print('SLP is ' + str(sum))
print(m)
print(m.rbf.lengthscale)
saveModel(m, name=('WarpedFinal//Final.npy'))
fullPlot(mean, quantiles, name=('Warped' + str(x) + '.png'), num = x, CCC=CCC)

