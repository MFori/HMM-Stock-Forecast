import numpy as np


class HMM:
    states: int
    pi: np.array  # start probabilities
    a: np.array  # transition probability matrix
    b: np.array  # emission probability matrix

    def __init__(self, states):
        self.states = states

    def init_params(self, X):
        N = self.states

        # data_for_clustering = [item for item in X]
        # data_for_clustering = np.array(data_for_clustering)
        # data_for_clustering = np.concatenate(data_for_clustering).reshape(-1, 1)

        # kmeans = KMeans(n_clusters=N, random_state=0)
        # y_means = kmeans.fit_predict(data_for_clustering)

        # X_ = [data_for_clustering[y_means == i] for i in range(N)]

        # distributions = []
        # for i in range(n_components):
        #     dist = NormalDistribution.blank()
        #     dist.fit(X_[i])
        #     distributions.append(dist)

        self.a = np.ones((N, N)) / N
        self.pi = np.zeros(N)
        self.pi[0] = 1
        self.b = np.random.rand(self.states, 4)  # TODO get shape from X

       # # Transition Probabilities
       # a = np.ones((2, 2))
       # a = a / np.sum(a, axis=1)

       # # Emission Probabilities
       # b = np.array(((1, 3, 5), (2, 4, 6)))
       # b = b / np.sum(b, axis=1).reshape((-1, 1))

       # # Equal Probabilities for the initial distribution
       # initial_distribution = np.array((0.5, 0.5))

       # self.a = a
       # self.b = b
       # self.pi = initial_distribution

    def fit(self, X):
        return self.baum_welch(X, self.a, self.b, self.pi, n_iter=100)

    def forward(self, V, a, b, initial_distribution):
        alpha = np.zeros((V.shape[0], a.shape[0]))
        #alpha = initial_distribution * b

        #for i in range(a.shape[0]):
        #    alpha[0][i] = initial_distribution[i] * b[i]
        alpha[0, :] = initial_distribution * b[:, V[0]]

        for t in range(1, V.shape[0]):
            for j in range(a.shape[0]):
                # Matrix Computation Steps
                #                  ((1x2) . (1x2))      *     (1)
                #                        (1)            *     (1)
                alpha[t, j] = alpha[t - 1] @ a[:, j] * b[j, V[t]]

        return alpha

    def backward(self, V, a, b):
        beta = np.zeros((V.shape[0], a.shape[0]))

        # setting beta(T) = 1
        beta[V.shape[0] - 1] = np.ones((a.shape[0]))

        # Loop in backward way from T-1 to
        # Due to python indexing the actual loop will be T-2 to 0
        for t in range(V.shape[0] - 2, -1, -1):
            for j in range(a.shape[0]):
                beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]) @ a[j, :]

        return beta

    def baum_welch(self, V, a, b, pi, n_iter=100):
        M = a.shape[0]
        T = len(V)

        for n in range(n_iter):
            ###estimation step
            alpha = self.forward(V, a, b, pi)
            beta = self.backward(V, a, b)

            xi = np.zeros((M, M, T - 1))
            for t in range(T - 1):
                # joint probab of observed data up to time t @ transition prob * emisssion prob as t+1 @
                # joint probab of observed data from time t+1
                denominator = (alpha[t, :].T @ a * b[:, V[t + 1]].T) @ beta[t + 1, :]
                for i in range(M):
                    numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                    xi[i, :, t] = numerator / denominator

            gamma = np.sum(xi, axis=1)
            ### maximization step
            a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

            # Add additional T'th element in gamma
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

            K = b.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                b[:, l] = np.sum(gamma[:, V == l], axis=1)

            b = np.divide(b, denominator.reshape((-1, 1)))

        return a, b
