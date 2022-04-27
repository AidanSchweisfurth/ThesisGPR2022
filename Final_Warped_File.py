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
########################################
# Warped Regression File
########################################
# this file optimizes a warped system using SCG and INverse Cumulitive Gaussian Warping Function
# features pulled from below featureDir variable
# features extracted using the extractFeatures function

### Example of Extracting Features###
# extractFeatures('train_1.wav', 'features/88/train_1.csv')
# extractFeatures('recordings/train_8.wav', featureDir + 'train_8.csv')

# ratingDir contains files for ratings of each file, these are pulled directly from RECOLA
# outputDir is directory where final files are saved


testRatingsFile = 'ratings_individual/arousal/train_2.csv'
outputDir = 'WarpedFinal/'
ratingsDir = 'ratings_individual/arousal/'
featureDir = 'features/Full Feature/'

def extractRating(file):
    vals = np.empty(shape=[0, 6])
    f = open(ratingsDir + file, 'rt')
    data = csv.reader(f)
    for row in data:
        x = row[0].split(';')
        val = x[1:]
        vals = np.vstack((vals, [val]))
    my_data = np.genfromtxt(featureDir + file, delimiter=',')
    f.close()
    return my_data, vals[101:]

def getInputs(file='train_'):
    #pulls all 9 files into single array for use
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
    #np.savetxt('features/Full Feature/features_dev.csv', features, delimiter=",")
    #np.savetxt('features/Full Feature/ratings_dev.csv', ratings.astype(float), delimiter=",")
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

def fullPlot(mean, quantiles, name="Warped", CCC=0, num = 0):
    #generates plot for all 9 output files with annotators included
    for i in range(1, 10):
        val = np.arange(4, 300.04, 0.04)
        plt.fill_between(val, quantiles[0][(i-1)*7401:i*7401], quantiles[1][(i-1)*7401:i*7401],
                        facecolor="powderblue", # The fill color
                        color='royalblue',       # The outline color
                        alpha=0.2)          # Transparency of the fill
        plt.plot(val, mean[(i-1)*7401:i*7401])
        ratings = getRatings(ratingsDir + 'dev_' + str(i) + '.csv')  
        for t in range(6):
            plt.plot(val, list(map(float, ratings[:, t])))

        plt.title("GPR Warped 6 Annotators F" + str(i))
        fig = plt.gcf()
        fig.set_size_inches((25, 10), forward=False)
        fig.savefig(outputDir + name + '_Dev_' + str(i) + '.png', dpi=500)
        plt.close()
        #plt.show()
    f = open(outputDir + 'Warped' + str(num) + '.txt', mode='w')
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

def MSE(predicted, actual, Y, name='MSE'):
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

        
########## Extract Features from files #############
X, Y = getInputs(file = "train_")
#X = np.genfromtxt('features/Full Feature/features_train.csv', delimiter=',')
#Y = np.genfromtxt('features/Full Feature/ratings_train.csv', delimiter=',')

testFeat, testRatingss = getInputs(file = "dev_")
#testFeat = np.genfromtxt('features/Full Feature/features_dev.csv', delimiter=',')
#testRatingss = np.genfromtxt('features/Full Feature/ratings_dev.csv', delimiter=',')

#take mean of annotators for testing
testRatings = np.mean(testRatingss.astype(np.float64), axis = 1)

########### Load Gaussian Process ###############
## load hyperparameters from file
m = loadFile(X, Y, 'WarpedFinal/Warped708.npy')
## generate new Gaussian Process
# m = WarpedModelSimple(X, Y, GPy.kern.RBF(88, 1, 100, ARD=True), initialize = False)

# x is starting number for files, set to 1 if new system desired, otherwise set to number after file title number
x = 709

######### Optimization Loop ###################

# system does ~600 iterations per loop then pauses to save the data and start again
# continues until convergence
while (m.optimize(optimizer = 'SCG', messages=True, max_iters=200).status == 'maxiter exceeded'):
    mean, variance = m.predict(testFeat)  
    quantiles = m.predict_quantiles(testFeat)

    #calc current CCC
    CCC = str(calculateCCC(mean.flatten().astype(np.float), testRatings.flatten().astype(np.float)))
    print("CCC is " + CCC)

    # calc Current MSE
    #MSE(mean, testRatings, np.std(testRatingss, axis=1))

    # Save Hyperparameters to txt file
    f = open(outputDir + 'Warped' + str(x) + '.txt', mode='w')
    f.write("Final CCC is " + CCC)
    f.write("\nValues:")
    f.write('\n' + str(m))
    f.write("\nLength Scales:")
    f.write('\n' + str(m.rbf.lengthscale))  
    f.write("\nInducing Inputs:")  
    f.write('\n' + str(m.inducing_inputs))
    f.close()

    # save in a reloadable format
    saveModel(m, name=(outputDir + 'Warped' + str(x) + '.npy'))

    x += 1
    

print('Finished')
print('------------------------------')
mean, variance = m.predict(testFeat)  
quantiles = m.predict_quantiles(testFeat)
CCC = str(calculateCCC(mean.flatten().astype(np.float), testRatings.flatten().astype(np.float)))
print("Final CCC is " + CCC)
saveModel(m, name=(outputDir + 'Final.npy'))
fullPlot(mean, quantiles, name=('Warped' + str(x)), num = x, CCC=CCC)

