import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def plotData(myX, newFig=False):
    if newFig:
        plt.figure(figsize=(8,6))
    plt.plot(myX[:,0],myX[:,1],'b+')
    plt.xlabel('Latency [ms]',fontsize=16)
    plt.ylabel('Throughput [mb/s]',fontsize=16)
    plt.grid(True)

def getGaussianParams(myX,useMultivariate = True):
    m = myX.shape[0]
    mu = np.mean(myX,axis=0)
    if not useMultivariate:
        sigma2 = np.sum(np.square(myX-mu),axis=0)/float(m)
        return mu, sigma2
    else:
        sigma2 = ((myX-mu).T.dot(myX-mu))/float(m)
        return mu, sigma2

def gaus(myX, mymu, mysig2):
    m = myX.shape[0]
    n = myX.shape[1]
    if np.ndim(mysig2) == 1:
        mysig2 = np.diag(mysig2)

    norm = 1./(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(mysig2)))
    myinv = np.linalg.inv(mysig2)
    myexp = np.zeros((m,1))
    for irow in range(m):
        xrow = myX[irow]
        myexp[irow] = np.exp(-0.5*((xrow-mymu).T).dot(myinv).dot(xrow-mymu))
    return norm*myexp


def plotContours(mymu, mysigma2, newFig=False, useMultivariate=True):
    delta = .5
    myx = np.arange(0, 30, delta)
    myy = np.arange(0, 30, delta)
    meshx, meshy = np.meshgrid(myx, myy)
    coord_list = [entry.ravel() for entry in (meshx, meshy)]
    points = np.vstack(coord_list).T
    myz = gaus(points, mymu, mysigma2)
    myz = myz.reshape((myx.shape[0], myx.shape[0]))

    if newFig: plt.figure(figsize=(6, 4))

    cont_levels = [10 ** exp for exp in range(-20, 0, 3)]
    mycont = plt.contour(meshx, meshy, myz, levels=cont_levels)

    plt.title('Gaussian Contours', fontsize=16)


def computeF1(predVec, trueVec):
    P, R = 0., 0.
    if float(np.sum(predVec)):
        P = np.sum([int(trueVec[x]) for x in range(predVec.shape[0]) \
                    if predVec[x]]) / float(np.sum(predVec))
    if float(np.sum(trueVec)):
        R = np.sum([int(predVec[x]) for x in range(trueVec.shape[0]) \
                    if trueVec[x]]) / float(np.sum(trueVec))

    return 2 * P * R / (P + R) if (P + R) else 0


def selectThreshold(myycv, mypCVs):
    nsteps = 1000
    epses = np.linspace(np.min(mypCVs), np.max(mypCVs), nsteps)

    bestF1, bestEps = 0, 0
    trueVec = (myycv == 1).flatten()
    for eps in epses:
        predVec = mypCVs < eps
        thisF1 = computeF1(predVec, trueVec)
        if thisF1 > bestF1:
            bestF1 = thisF1
            bestEps = eps

    print("Best F1 is %f, best eps is %0.4g." % (bestF1, bestEps))
    return bestF1, bestEps

def plotAnomalies(myX, mybestEps, newFig = False, useMultivariate = True):
    ps = gaus(myX, *getGaussianParams(myX, useMultivariate))
    anoms = np.array([myX[x] for x in range(myX.shape[0]) if ps[x] < mybestEps])
    if newFig: plt.figure(figsize=(6,4))
    plt.scatter(anoms[:,0],anoms[:,1], s=80, facecolors='none', edgecolors='r')


mat=sio.loadmat('data/ex8data1.mat')
X=mat['X']
ycv=mat['yval']
Xcv=mat['Xval']

# plotData(X)
# plt.show()
mu, sig2 = getGaussianParams(X, useMultivariate = True)

plotData(X, newFig=True)
useMV = False
plotContours(*getGaussianParams(X, useMV), newFig=False, useMultivariate = useMV)
plt.show()

# Then contours with multivariate gaussian:


pCVs = gaus(Xcv, mu, sig2)
bestF1, bestEps = selectThreshold(ycv,pCVs)

plotData(X, newFig=True)
plotContours(mu, sig2, newFig=False, useMultivariate=True)
plotAnomalies(X, bestEps, newFig=False, useMultivariate=True)
plt.show()
