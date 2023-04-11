import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import pandas as pd
import time 

def Initialize_parameters(df, k, n_no_of_samples, m_no_of_features):

    means = np.random.randn(k, m_no_of_features)
    covariances = np.array([np.eye(m_no_of_features)] * k)
    priors = np.ones(k)/k

    idx_of_clusters = np.random.randint(k, size = n_no_of_samples)

    # for j in range (k):
    #     df_clusters = df[idx_of_clusters==j]
    #     means[j] = np.mean(df_clusters, axis = 0)
    #     covariances[j] = np.cov(df_clusters.T)
    #     priors[j] = df_clusters.shape[0]/n_no_of_samples

    prob = np.zeros((n_no_of_samples, k))

    return means, covariances, priors, prob

def get_pdf(df, means, covariances):
        
        return multivariate_normal.pdf(df, means, covariances, allow_singular=True)

df = pd.read_csv('data6D.txt', sep= ' ', header=None).values

n_no_of_samples = df.shape[0]
m_no_of_features = df.shape[1]

k_max = 10
max_iterations = 1500

log_likelihoods = np.zeros(k_max)

for k in range(1, k_max+1):

    start_time = time.time()

    means, covariances, priors, prob = Initialize_parameters(df, k, n_no_of_samples, m_no_of_features)

    log_likelihood = 0.0

    for itr in range(max_iterations):

        # E-step
        
        pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]

        for j in range(k):
            prob[:, j] = priors[j] * pdfs[j]

        prob = prob / np.sum(prob, axis=1).reshape(n_no_of_samples, 1)

        # M-step

        sum_prob = []

        for j in range(k):
            sum_prob = np.sum(prob[:, j])
            priors[j] = sum_prob / n_no_of_samples

            means[j] = np.sum(prob[:, j].reshape(n_no_of_samples, 1) * df, axis=0) / sum_prob
            means[j] += 1e-6

            covariances[j] = np.sum(prob[:, j].reshape(n_no_of_samples, 1, 1) * (df - means[j]).reshape(n_no_of_samples, 1, m_no_of_features) * (df - means[j]).reshape(n_no_of_samples, m_no_of_features, 1), axis=0) / sum_prob
            covariances[j] += 1e-6

        log_likelihood_temp = 0.0

        pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]

        log_likelihood_temp = np.sum(np.log(np.sum([priors[j] * pdfs[j] for j in range(k)], axis=0)))
        log_likelihoods[k-1] = log_likelihood_temp

        if np.abs(log_likelihood_temp - log_likelihood) < 1e-5:
            print('k = ', k, ' converged in ', itr, ' iterations')
            print('log_likelihood = ', log_likelihood_temp)
            print('time taken = ', time.time() - start_time)
            break

        if(itr == max_iterations-1):
            print('k = ', k, ' did not converge')
            print('log_likelihood = ', log_likelihood_temp)
            print('time taken = ', time.time() - start_time)

        log_likelihood = log_likelihood_temp

plt.plot(range(1, k_max+1), log_likelihoods)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Converged log-likelihood')
plt.title('Number of clusters vs Converged log-likelihood m='+ str(m_no_of_features))
plt.show()

k = input('Enter the number of clusters: ')
k = int(k)

if(m_no_of_features >2):
    pca = PCA(n_components=2)
    df = pca.fit_transform(df)
    m_no_of_features = 2

means, covariances, priors, prob = Initialize_parameters(df, k, n_no_of_samples, m_no_of_features)

log_likelihood = 0.0

plt.ion()

for itr in range(max_iterations):
    
    # E-step

    pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]

    for j in range(k):
        prob[:, j] = priors[j] * pdfs[j]

    prob = prob / np.sum(prob, axis=1).reshape(n_no_of_samples, 1)

    # M-step

    sum_prob = []

    for j in range(k):
        sum_prob = np.sum(prob[:, j])
        priors[j] = sum_prob / n_no_of_samples

        means[j] = np.sum(prob[:, j].reshape(n_no_of_samples, 1) * df, axis=0) / sum_prob
        means[j] += 1e-6

        covariances[j] = np.sum(prob[:, j].reshape(n_no_of_samples, 1, 1) * (df - means[j]).reshape(n_no_of_samples, 1, m_no_of_features) * (df - means[j]).reshape(n_no_of_samples, m_no_of_features, 1), axis=0) / sum_prob
        covariances[j] += 1e-6

    log_likelihood_temp = 0.0

    pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]

    log_likelihood_temp = np.sum(np.log(np.sum([priors[j] * pdfs[j] for j in range(k)], axis=0)))

    if np.abs(log_likelihood_temp - log_likelihood) < 1e-5 or itr == max_iterations-1:
        plt.title('Final Clusters')
        break

    log_likelihood = log_likelihood_temp

    plt.clf()
    plt.scatter(df[:, 0], df[:, 1], c=prob.argmax(axis=1), cmap=cm.jet, s=40, edgecolor='k', alpha=0.7)
    #plt.scatter(df[:, 0], df[:, 1])
    x, y = np.mgrid[np.min(df[:, 0]):np.max(df[:, 0]):.01, np.min(df[:, 1]):np.max(df[:, 1]):.01]

    df_reprocessed = np.dstack((x, y))

    for j in range(k):
        rv = get_pdf(df_reprocessed, means[j], covariances[j])
        plt.contour(x, y, rv, colors='blue', alpha=0.8, linewidths=1)

    plt.title('Clustered data points at iteration ' + str(itr))
    plt.draw()
    plt.pause(0.05)

print("Final Clusters Plotted")
plt.ioff()
plt.show()