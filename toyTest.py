import GPy
import numpy as np
import matplotlib.pyplot as plt
from warped_model_Annotators import WarpedModelSimple

def fullPlot(mean, quantiles, name="Plot.png", Y=None):
    val = np.arange(0, 20)
    #x = range(1, 30)/10
    #print(x)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.fill_between(val, quantiles[0], quantiles[1],
    #plt.fill_between(val, quantiles[0], quantiles[1],
                    facecolor="powderblue", # The fill color
                    color='royalblue',       # The outline color
                    alpha=0.2)          # Transparency of the fill
    plt.plot(val, mean)  
    plt.plot(val, list(map(float, Y)))
    plt.title("GPR Warped 6 Annotators")
    plt.show()

def funct(n=20, noise_variance=1e-3):
    vals = np.zeros((n, 1))
    for x in range(0, n):
        vals[x] = x
    X = np.random.uniform(0., 3., (n, 1))

    Y = X/3 * np.sin(X) + np.random.randn(n, 1) * noise_variance**0.5
    Y_out1 = np.zeros((n, 1))
    for x in range(len(Y)):
        Y_out1[x] = np.array([Y[x]])
    Y = X/3 * np.sin(X) + np.random.randn(n, 1) * noise_variance**0.5
    Y_out2 = np.zeros((n, 1))
    for x in range(len(Y)):
        Y_out2[x] = np.array([Y[x]])
    Y = X/3 * np.sin(X) + np.random.randn(n, 1) * noise_variance**0.5
    Y_out3 = np.zeros((n, 1))
    for x in range(len(Y)):
        Y_out3[x] = np.array([Y[x]])
    Y = X/3 * np.sin(X) + np.random.randn(n, 1) * noise_variance**0.5
    Y_out4 = np.zeros((n, 1))
    for x in range(len(Y)):
        Y_out4[x] = np.array([Y[x]])
    Y = X/3 * np.sin(X) + np.random.randn(n, 1) * noise_variance**0.5
    Y_out5 = np.zeros((n, 1))
    for x in range(len(Y)):
        Y_out5[x] = np.array([Y[x]])
    Y = X/3 * np.sin(X) + np.random.randn(n, 1) * noise_variance**0.5
    Y_out6 = np.zeros((n, 1))
    for x in range(len(Y)):
        Y_out6[x] = np.array([Y[x]])
    Y_out = np.hstack((Y_out1, Y_out2, Y_out3, Y_out4, Y_out5, Y_out6))
    return X, Y_out


X1, Y = funct()
#print(Y)
kernel = GPy.kern.RBF(1, 100)
m = WarpedModelSimple(X1, Y, kernel)
m.optimize(optimizer = 'SCG', max_iters=200, messages = True)
vals = np.zeros((20, 1))
for x in range(0, 20):
    vals[x] = x

X2 = np.random.uniform(1., 3., (20, 1))
mean, var = m.predict(X2)
Y = X2/3 * np.sin(X2)
fullPlot(mean, m.predict_quantiles(X2), Y=Y)