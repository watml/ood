import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
import seaborn as sns
import math
import torch
import torchvision.transforms as T

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve

def modify_inf(result):
	# change nan to zero
	result[result!=result]=0

	# change inf to 10**20
	rsize = len(result)
	
	for i in range(0,rsize):
		
		if np.isinf(result[i]):
			if result[i]>0:
				result[i] = +10**20
			else:
				result[i] = -10**20
	return result

def modify_abn(z, th):
    # keep values above th
    pos = np.where(z >= th)[0]
    z = z[pos]
    return z

def get_q_array(q):
    q_arr = np.array([])
    for i in range(len(q)):
        q_arr = np.append(q_arr, np.array(q[i]))
    return q_arr

def get_u_array(u):
    u_arr = u[0]
    for i in range(1,len(u)):
        u_arr = np.vstack((u_arr, u[i]))
    return u_arr


def plotROC(sample_test_ind, sample_test_ood):
    y_test_ind = np.zeros(sample_test_ind.shape[0])  # lower scores indicate more normal / higher -> more novel
    y_test_ood = np.ones(sample_test_ood.shape[0])
    y_true = np.append(y_test_ind, y_test_ood)

    sample_score = np.append(sample_test_ind, sample_test_ood)
    sample_score = (sample_score - np.min(sample_score)) / (np.max(sample_score) - np.min(sample_score))
    fpr, tpr, _ = roc_curve(y_true, sample_score)
    AUC = roc_auc_score(y_true, sample_score)
    plt.plot(fpr, tpr, label='AUC = %0.3f' % AUC)
    AUPR = average_precision_score(y_true, sample_score)
    print('AUROC: ', AUC)
    print('AUPR: ', AUPR)
    plt.legend()


def plotHist(sample_train, sample_test_ind, sample_test_ood, InD, OOD, sample_name, nbin1=50, nbin2=50, nbin3=50):
    # plot with three colors
    w_train = np.ones_like(sample_train) / sample_train.shape[0]
    w_test_ind = np.ones_like(sample_test_ind) / sample_test_ind.shape[0]
    w_test_ood = np.ones_like(sample_test_ood) / sample_test_ood.shape[0]
    plt.hist(sample_train, bins=nbin1, label=InD + '_train', alpha=0.6, weights=w_train)
    plt.hist(sample_test_ind, bins=nbin2, label=InD + '_test', alpha=0.6, weights=w_test_ind)
    plt.hist(sample_test_ood, bins=nbin3, label=OOD + '_test', alpha=0.6, weights=w_test_ood)
    plt.xlabel(sample_name, fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=18)


def plotHist2(sample_test_ind, sample_test_ood, InD, OOD, sample_name, nbin2=20, nbin3=20):
    # same as above, but only with two colors
    w_test_ind = np.ones_like(sample_test_ind) / sample_test_ind.shape[0]
    w_test_ood = np.ones_like(sample_test_ood) / sample_test_ood.shape[0]
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.hist(sample_test_ind, bins=nbin2, label=InD + '_test', alpha=0.6, weights=w_test_ind, color=cycle[1])
    plt.hist(sample_test_ood, bins=nbin3, label=OOD + '_test', alpha=0.6, weights=w_test_ood, color=cycle[2])
    plt.xlabel(sample_name)
    plt.legend()





def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) *
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

# Plot bivariate distribution
def generate_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = 100 # grid size
    x1s = np.linspace(-10, 10, num=nb_of_x)
    x2s = np.linspace(-10, 10, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i,j] = multivariate_normal(
                np.matrix([[x1[i,j]], [x2[i,j]]]),
                d, mean, covariance)
    return x1, x2, pdf  # x1, x2, pdf(x1,x2)

def plotGaussian(mean1, mean2):
    # sns.set_style('darkgrid')
    mean_ind = np.matrix([[mean1],[mean1]])
    x1, y1, p1 = generate_surface(mean_ind, np.eye(2), 2)

    mean_ood = np.matrix([[mean2], [mean2]])
    x2, y2, p2 = generate_surface(mean_ood, np.eye(2), 2)

    # Plot bivariate distribution
    plt.contour(x1, y1, p1, 10, cmap=cm.jet)
    plt.contour(x2, y2, p2, 10, cmap=cm.jet)
    plt.xlabel('$x_0$', fontsize=54)
    plt.ylabel('$x_1$', fontsize=54)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.axis([-3, 8, -3, 8])

def plotGaussian2(mean1, mean2, range=10, cov1=np.eye(2), cov2=np.eye(2)):
    # sns.set_style('darkgrid')
    mean_ind = np.matrix([[mean1[0]],[mean1[1]]])
    cov1 = np.matrix([[cov1[0], cov1[1]],[cov1[2], cov1[3]]])
    x1, y1, p1 = generate_surface(mean_ind, cov1, 2)

    mean_ood = np.matrix([[mean2[0]], [mean2[1]]])
    cov2 = np.matrix([[cov2[0], cov2[1]], [cov2[2], cov2[3]]])
    x2, y2, p2 = generate_surface(mean_ood, cov2, 2)

    # Plot bivariate distribution
    plt.contour(x1, y1, p1, 10, cmap=cm.jet)
    plt.contour(x2, y2, p2, 10, cmap=cm.jet)
    plt.xlabel('$x_0$', fontsize=13)
    plt.ylabel('$x_1$', fontsize=13)
    plt.axis([-range, range, -range, range])

def KLD(mean, cov, n):
    # equ.(4) in the reference
    # return 0.5 * (- np.log(np.linalg.det(cov)) + np.trace(cov) + np.matmul(np.transpose(mean), mean) - n)
    return 0.5 * (np.matmul(np.transpose(mean), mean))

def cramervonmises(rvs, cdf, args=()):
    """
    Adapted from https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/_hypotests.py#L266-L379
    """
    if cdf=='norm':
        cdf = stats.norm.cdf

    vals = np.sort(np.asarray(rvs))

    if vals.size <= 1:
        raise ValueError('The sample must contain at least two observations.')
    if vals.ndim > 1:
        raise ValueError('The sample must be one-dimensional.')

    n = len(vals)
    cdfvals = cdf(vals, *args)

    u = (2*np.arange(1, n+1) - 1)/(2*n)
    w = 1/(12*n) + np.sum((u - cdfvals)**2)
    return w

def cramervonmises_2samp(x, y):
    """
    Adapted from https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/_hypotests.py#L1288-L1439
    """
    xa = np.sort(np.asarray(x))
    ya = np.sort(np.asarray(y))

    if xa.size <= 1 or ya.size <= 1:
        raise ValueError('x and y must contain at least two observations.')
    if xa.ndim > 1 or ya.ndim > 1:
        raise ValueError('The samples must be one-dimensional.')

    nx = len(xa)
    ny = len(ya)


    # get ranks of x and y in the pooled sample
    z = np.concatenate([xa, ya])
    # in case of ties, use midrank (see [1])
    r = stats.rankdata(z, method='average')
    rx = r[:nx]
    ry = r[nx:]

    # compute U (eq. 10 in [2])
    u = nx * np.sum((rx - np.arange(1, nx+1))**2)
    u += ny * np.sum((ry - np.arange(1, ny+1))**2)

    # compute T (eq. 9 in [2])
    k, N = nx*ny, nx + ny
    t = u / (k*N) - (4*k - 1)/(6*N)

    return t