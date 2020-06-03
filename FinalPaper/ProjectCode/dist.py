import scipy.special as ss
from scipy.stats import beta, norm, uniform, gamma, multivariate_normal
from scipy.linalg import svd
from numpy.random import randn
import numpy as np
import copy

def beta_dist(mu, sig):
    a = (1-mu) * mu**2 / sig**2 - mu
    b  = a * (1/mu -1)
    dist  = beta(a, b)
    dist.name = 'beta'
    return dist

def gamma_dist(mu, sig):
    b  = sig**2 / mu
    a = mu / b
    dist  = gamma(a, scale=b)
    dist.name = 'gamma'
    return dist

def inv_gamma_dist(mu, sig):
    dist = InvGamma(mu, sig)
    dist.name = 'inv_gamma'
    return dist

def norm_dist(mu, sig):
    dist = norm(loc=mu, scale=sig)
    dist.name = 'norm'
    return dist

def unif_dist(mu, sig):
    dist = uniform(loc=mu, scale=(sig-mu))
    dist.name = 'uniform'
    return dist

class InvGamma(object):
    def __init__(self, v, t):
        self.v = v
        self.t = t

    def pdf(self, x):
        v = self.v
        t = self.t
        pdf = 2 * (v*t**2/2)**(v/2) * np.exp((-v*t**2)/(2*x**2)) / \
                        ss.gamma(v/2) / x**(v+1)

        return pdf

    def logpdf(self, x):
        v = self.v #v
        t = self.t #t
        if x < 0:
            return -np.inf

        log_pdf = np.log(2) - np.log(ss.gamma(v/2)) + (v/2)*np.log(v*t**2/2) \
                        - ((v+1)/2)*np.log(x**2) - v*t**2/(2*x**2)

        return log_pdf

    def rvs(self,random_state=None):
        v = self.v #v
        t = self.t #t
        return np.sqrt(v * t**2 / sum(np.random.randn(round(v))**2))


class DegenerateMvNormal(object):
    def __init__(self, mu, sig, sig_inv=np.array([]), lamb_vals=np.array([])):
        self.mu        = mu
        self.sig       = sig
        self.sig_inv   = sig_inv
        self.lamb_vals = lamb_vals

    def rvs(self, cc=1.0):
        return self.mu + cc*self.sig @ randn(len(self.mu))

    def logpdf(self, x):
        if len(self.sig_inv) == 0:
            self.sig_inv   = np.linalg.pinv(self.sig)
            lamb_all       = np.linalg.eigvals(self.sig)
            self.lamb_vals = np.array([l for l in lamb_all if l > 1e-6])

        return -(len(self.mu) * np.log(2*np.pi) + sum(np.log(self.lamb_vals)) + ((x-self.mu).T @ self.sig_inv @ (x-self.mu)))/2.0

def init_deg_mvnormal(mu, sig):
    U, lamb_vals, Vt = svd(sig, full_matrices=False)
    lamb_inv = [1/lamb if lamb > 1e-6 else 0.0 for lamb in lamb_vals]
    sig_inv  = Vt.T @ np.diag(lamb_inv) @ U.T

    return DegenerateMvNormal(mu, sig, sig_inv, lamb_vals)


class MvNormal(object):
    def __init__(self, mu, sig):
        self.mu  = mu
        self.sig = sig
        self.MV  = multivariate_normal(mu, sig)

    def rvs(self, random_state=None):
        return self.MV.rvs(random_state=random_state)

    def pdf(self, x):
        return self.MV.pdf(x)

    def logpdf(self, x):
        return self.MV.logpdf()

class Mix3MVmodel(object):
    def __init__(self, dist1, dist2, dist3, W):
        self.dist1 = dist1
        self.dist2 = dist2
        self.dist3 = dist3
        self.W     = W

    def rvs(self,random_state=np.array([None,None,None])):
        draw = np.array([self.dist1.rvs(random_state=random_state[0]),self.dist2.rvs(random_state=random_state[1]),self.dist3.rvs(random_state=random_state[2])])
        idx  = np.random.choice(np.arange(3), replace=False, p=self.W)
        vals = draw[idx]
        return vals
