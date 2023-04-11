"""If the number of attributes is equal to 2, 
you need to show plots of estimated GMM. 
After each iteration (an E-step and an M-step), 
plot the data points and gaussian distributions in a 2D plot.
Do not save the plots to a file. 
After running the EM algorithm, 
the plot should update as the algorithm advances"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import pandas as pd

def get_pdf(df, means, covariances):
        
        if(np.isclose(means, 0.0).any()):
            # print("Means are zero")
            means += 1e-6
    
        if(np.isclose(covariances, 0.0).any()):
            # print("covariances are zero")
            covariances += 1e-6
    
        return multivariate_normal.pdf(df, means, covariances, allow_singular=True)

# Load data
df = pd.read_csv('data6D.txt', sep= ' ', header=None)
df = df.values

n_no_of_samples = df.shape[0]
m_no_of_features = df.shape[1]

if(m_no_of_features > 2):
    pca = PCA(n_components=2)
    df = pca.fit_transform(df)
    m_no_of_features = 2

k = 5
max_iterations = 100

means = np.random.randn(k, m_no_of_features)

covariances = np.array([np.eye(m_no_of_features)] * k)

priors = np.ones(k)/k

y = np.random.randint(k, size = n_no_of_samples)
for j in range (k):
    df_j = df[y==j]
    means[j] = np.mean(df_j, axis = 0)
    covariances[j] = np.cov(df_j.T)
    priors[j] = df_j.shape[0]/n_no_of_samples

prob = np.zeros((n_no_of_samples, k))

log_likelihood = 0.0


plt.ion()
for itr in range(max_iterations):
    # E-step

    pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]
    
    for i in range(n_no_of_samples):
        for j in range(k):
            prob[i, j] = priors[j] * pdfs[j][i]
        prob[i] /= np.sum(prob[i])

    # M-step
    sum_prob = []

    for j in range(k):
        sum_prob = np.sum(prob[:, j])
        priors[j] = sum_prob / n_no_of_samples

        means[j] = 0.0
        for i in range(n_no_of_samples):
                means[j] += prob[i, j] * df[i]

        means[j] /= sum_prob

        covariances[j] = 0.0
        for i in range(n_no_of_samples):
            covariances[j] += prob[i, j] * np.outer(df[i] - means[j], df[i] - means[j])
            #covariances[j] += prob[i , j] * (df[i] - means[j]).reshape(m_no_of_features, 1) * (df[i] - means[j]).reshape(1, m_no_of_features)
        covariances[j] /= np.sum(prob[:, j])

    # means+=1e-6
    # covariances+=1e-6

    # Compute the log-likelihood
    log_likelihood_temp = 0.0

    pdfs = [get_pdf(df, means[j], covariances[j]) for j in range(k)]

    for i in range(n_no_of_samples):
        log_likelihood_temp+=np.log(np.sum([priors[j] * pdfs[j][i] for j in range(k)]))
        
    if np.abs(log_likelihood_temp - log_likelihood) < 1e-3 or itr == max_iterations-1:
        plt.title("final plot")
        break
    log_likelihood = log_likelihood_temp

    #print("plotting at iteration " + str(itr))

    plt.clf()
    plt.scatter(df[:, 0], df[:, 1] , c=prob.argmax(axis=1), cmap='viridis', s=40, edgecolor='k', alpha=0.5)
    x, y = np.mgrid[np.min(df[:, 0]):np.max(df[:, 0]):.01, np.min(df[:, 1]):np.max(df[:, 1]):.01]
    positions = np.dstack((x, y))
    for j in range(k):
        rv = get_pdf(positions, means[j], covariances[j])
        plt.contour(x, y, rv, colors='black', alpha=0.5, linewidths=1)
    plt.title("Clustered data points at iteration " + str(itr))
    plt.draw()
    plt.pause(0.001)

print("plotting done")
plt.ioff()
plt.show()