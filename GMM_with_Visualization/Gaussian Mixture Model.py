import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import pandas as pd
import time 

def Initialize_parameters(df, k, n_no_of_samples, m_no_of_features):
    # Initialize the means with random between min and max values of each feature
    # bmeans = np.zeros((k, m_no_of_features))
    # for i in range(m_no_of_features):
    #     min_feature = min(df[:,i])
    #     max_feature = max(df[:,i])
    #     means[:,i] = np.random.uniform(min_feature, max_feature, k)
    means = np.random.randn(k, m_no_of_features)

    # #pick k random points from the dataset as the initial means
    # means = df[np.random.choice(n_no_of_samples, k, replace=False)]
    # print(means)

    # Initialize the covariance matrices
    covariances = np.array([np.eye(m_no_of_features)] * k)

    # Initialize the mixing coefficients
    priors = np.ones(k)/k

    idx_of_clusters = np.random.randint(k, size = n_no_of_samples)

    # print(idx_of_clusters)

    # for j in range (k):
    #     df_clusters = df[idx_of_clusters==j]
    #     means[j] = np.mean(df_clusters, axis = 0)
    #     covariances[j] = np.cov(df_clusters.T)
    #     priors[j] = df_clusters.shape[0]/n_no_of_samples

    # Initialize the Probabilities
    prob = np.zeros((n_no_of_samples, k))

    return means, covariances, priors, prob

def get_pdf(df, means, covariances):
    
    # if(np.isclose(means, 0.0).any()):
    #     # print("Means are zero")
    #     means += 1e-6

    # if(np.isclose(covariances, 0.0).any()):
    #     # print("covariances are zero")
    #     covariances += 1e-6

    return multivariate_normal.pdf(df, means, covariances, allow_singular=True)

# Load data
df = pd.read_csv('data2Dnew.txt', sep= ' ', header=None)
df = df.values

n_no_of_samples = df.shape[0]
m_no_of_features = df.shape[1]

# take input of k
#k_max  = int(input("Enter the number of maximum clusters: "))
k_max = 10

# take input of maximum iterations
#max_iterations = int(input("Enter the number of maximum iterations: "))
max_iterations = 100

log_likelihoods = np.zeros(k_max)

#For each value of k
#Apply the EM algorithm to estimate the GMM.
#Keep a note of the converged log-likelihod.

for k in range(1, k_max+1):
    #calculate time
    start_time = time.time()
    # Initialize the parameters

    means, covariances, priors, prob = Initialize_parameters(df, k, n_no_of_samples, m_no_of_features)

    # Initialize the log-likelihood
    log_likelihood = 0.0

    # Iterate until convergence
    for itr in range(max_iterations):
        # E-step
        # Calculate the Probabilities

        pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]
        # for i in range(n_no_of_samples):
        #     for j in range(k):
        #         prob[i, j] = priors[j] * pdfs[j][i]
        #     prob[i] = prob[i] / np.sum(prob[i])

        for j in range(k):
            prob[:, j] = priors[j] * pdfs[j]

        prob = prob / np.sum(prob, axis=1).reshape(n_no_of_samples, 1)
        # M-step
        sum_prob = []

        for j in range(k):
            # Calculate the priors
            sum_prob = np.sum(prob[:, j])
            priors[j] = sum_prob / n_no_of_samples

        # Calculate the means
            means[j] = 0.0
            # for i in range(n_no_of_samples):
            #     means[j] += prob[i, j] * df[i]
            means[j] = np.sum(prob[:, j].reshape(n_no_of_samples, 1) * df, axis=0)

            means[j] /= sum_prob
            means[j] += 1e-6

        # Calculate the covariance matrices
            covariances[j] = 0.0
            # for i in range(n_no_of_samples):
            #     covariances[j] += prob[i, j] * np.outer(df[i] - means[j], df[i] - means[j])
            #     #covariances[j] += prob[i , j] * (df[i] - means[j]).reshape(m_no_of_features, 1) * (df[i] - means[j]).reshape(1, m_no_of_features)
            # covariances[j] /= np.sum(prob[:, j])
            covariances[j] = np.sum(prob[:, j].reshape(n_no_of_samples, 1, 1) * (df - means[j]).reshape(n_no_of_samples, 1, m_no_of_features) * (df - means[j]).reshape(n_no_of_samples, m_no_of_features, 1), axis=0)
            covariances[j] /= sum_prob
            covariances[j] += 1e-6

            
        # Calculate the log-likelihood
        log_likelihood_temp = 0.0

        pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]
        # for i in range(n_no_of_samples):

        #     #log_likelihood_temp+=np.log(np.sum([priors[j] * multivariate_normal.pdf(df[i], means[j], covariances[j], allow_singular=True) for j in range(k)]))
        #     log_likelihood_temp+=np.log(np.sum([priors[j] * pdfs[j][i] for j in range(k)]))
        log_likelihood_temp = np.sum(np.log(np.sum([priors[j] * pdfs[j] for j in range(k)], axis=0)))
        log_likelihoods[k-1] = log_likelihood_temp

        # Check for convergence
        if np.abs(log_likelihood_temp - log_likelihood) < 1e-3:
            print('k =', k)
            print('Converged at iteration', itr)
            print('Log-likelihood', log_likelihood_temp)
            end_time = time.time()
            print('Time taken', end_time - start_time)
            break
        if(itr == max_iterations-1):
            print('k =', k)
            print('Not converged')
            print('Log-likelihood', log_likelihood_temp)
            end_time = time.time()
            print('Time taken', end_time - start_time)
        log_likelihood = log_likelihood_temp


plt.plot(range(1, k_max+1), log_likelihoods)
plt.xlabel('Number of Components (k)')
plt.ylabel('Converged Log-Likelihood')
plt.title("number of components vs log-likelihood")
plt.show()

# Choose the best value for k based on the plot
k = input("Enter the number of clusters: ")
k = int(k)

if(m_no_of_features > 2):
    pca = PCA(n_components=2)
    df = pca.fit_transform(df)
    m_no_of_features = 2

means, covariances, priors, prob = Initialize_parameters(df, k, n_no_of_samples, m_no_of_features)

log_likelihood = 0.0

plt.ion()
for itr in range(max_iterations):
    # E-step

    pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]
    
    # for i in range(n_no_of_samples):
    #     for j in range(k):
    #         prob[i, j] = priors[j] * pdfs[j][i]
    #     prob[i] /= np.sum(prob[i])

    for j in range(k):
        prob[:, j] = priors[j] * pdfs[j]

    prob = prob / np.sum(prob, axis=1).reshape(n_no_of_samples, 1)

    # M-step
    sum_prob = []

    for j in range(k):
        sum_prob = np.sum(prob[:, j])
        priors[j] = sum_prob / n_no_of_samples

        means[j] = 0.0
        # for i in range(n_no_of_samples):
        #         means[j] += prob[i, j] * df[i]
        means[j] = np.sum(prob[:, j].reshape(n_no_of_samples, 1) * df, axis=0)
        means[j] /= sum_prob
        means[j] += 1e-6

        covariances[j] = 0.0
        # for i in range(n_no_of_samples):
        #     #covariances[j] += prob[i, j] * np.outer(df[i] - means[j], df[i] - means[j])
        #     covariances[j] += prob[i , j] * (df[i] - means[j]).reshape(m_no_of_features, 1) * (df[i] - means[j]).reshape(1, m_no_of_features)
        covariances[j] = np.sum(prob[:, j].reshape(n_no_of_samples, 1, 1) * (df - means[j]).reshape(n_no_of_samples, 1, m_no_of_features) * (df - means[j]).reshape(n_no_of_samples, m_no_of_features, 1), axis=0)
        covariances[j] /= np.sum(prob[:, j])
        covariances[j] += 1e-6

    # Compute the log-likelihood
    log_likelihood_temp = 0.0

    pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]

    # for i in range(n_no_of_samples):
    #     log_likelihood_temp+=np.log(np.sum([priors[j] * pdfs[j][i] for j in range(k)]))
    log_likelihood_temp = np.sum(np.log(np.sum([priors[j] * pdfs[j] for j in range(k)], axis=0)))
        
    if np.abs(log_likelihood_temp - log_likelihood) < 1e-3 or itr == max_iterations-1:
        plt.title("Final Plot")
        break
    log_likelihood = log_likelihood_temp

    #print("plotting at iteration " + str(itr))

    plt.clf()
    plt.scatter(df[:, 0], df[:, 1] , c=prob.argmax(axis=1), cmap='viridis', s=40, edgecolor='k', alpha=0.5)
    x, y = np.mgrid[np.min(df[:, 0]):np.max(df[:, 0]):.01, np.min(df[:, 1]):np.max(df[:, 1]):.01]

    df_reprocessed = np.dstack((x, y))

    for j in range(k):
        rv = get_pdf(df_reprocessed, means[j], covariances[j])
        plt.contour(x, y, rv, colors='blue', alpha=0.8, linewidths=1)
        # plt.contour(x, y, rv, colors=[cm.rainbow(j/k)], alpha=0.9, linewidths=1)
        # plt.contour(x, y, rv, alpha=0.9, linewidths=1)
    plt.title("Clustered data points at iteration " + str(itr))
    plt.draw()
    plt.pause(0.01)

print("plotting done")
plt.ioff()
plt.show()