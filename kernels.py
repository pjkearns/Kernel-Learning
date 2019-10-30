import numpy as np

def linear(x,y):
    """Linear Kernel"""
    return np.dot(x,y)
    
def polynomial(x,y,r=1,n=2):
    """Polynomial Kernel"""
    return (r + np.dot(x,y))**n
    
def rbf(x,y,sig=1):
    """RBF (Gaussian) Kernel"""
    diff = np.subtract(x,y)
    return np.exp(-np.dot(diff,diff)/(2*sig**2))
    
def tanh(x,y,kap=1,tht=0.5):
    """Hyperbolic Tangent Kernel"""
    return np.tanh(kap*np.dot(x,y)+tht)