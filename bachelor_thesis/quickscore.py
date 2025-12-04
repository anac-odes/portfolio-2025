#!/usr/bin/env python
# coding: utf-8

# This file only contains the quickscore algorithm - standard and gmpy2 version - for easier usage in other notebooks

# # Standard Quickscore

# In[1]:


def fun_quickscoreW2(prev, pfmin, pfminneg):
    """
    Heckerman's Quickscore function
    
    Arguments:
    - prev: Prevalences: p(d_i=1)
    - pfmin: 1 - sensitivity (for positive tests): 1 - p(fj|only d_i =1), fj in F+
    - pfminneg: Products of 1 - sensitivity for negatives: Prod(1 - p(fj|only d_i =1)), product taken over j with fj in F-
    
    Returns:
    - posterior: Array of P(d_i=1|F+,F-) ... the most important outcome ...
    - pfpluspd: p(F+,F-) and P(F+,F-|d_i=1)p(d_i=1)  ... for debugging etc. ...
    - pfplus: p(F+,F-) and P(F+,F-|d_i=1).   ... for debugging etc. ...
    """
    prev = np.array(prev).reshape(-1)  # Ensure prev is a 1D array
    pfminneg = np.array(pfminneg).reshape(-1)
    n = len(prev)

    if pfmin is not None:
        m, n_pfmin = pfmin.shape
        if n_pfmin != n:
            raise ValueError('inconsistent n')
    else:
        m = 0
    
    pfplus = np.zeros(n + 1)
    prevend = np.ones(n + 1)
    prevend[1:] = prev
    
    if pfminneg is not None:
        prevminneg = prev * pfminneg
    else:
        prevminneg = prev
    
    v = np.zeros(m, dtype=int)
    ready = False
    while not ready:
        myset = np.where(v == 1)[0]
        if len(myset) == 0:
            Xi = np.prod(prevminneg + (1 - prev))
            Di = np.concatenate(([1], pfminneg / (prevminneg + (1 - prev))))
            pfplus += Xi * Di
        elif len(myset) == 1:
            Xi = np.prod(pfmin[myset, :] * prevminneg + (1 - prev))
            Di = np.concatenate(([1], ((pfmin[myset, :] * pfminneg) / (pfmin[myset, :] * prevminneg + (1 - prev))).reshape(-1,)))
            pfplus += ((-1) ** len(myset)) * Xi * Di
        else:
            prodpfmin = np.prod(pfmin[myset, :], axis=0)
            Xi = np.prod(prodpfmin * prevminneg + (1 - prev))
            Di = np.concatenate(([1], ((prodpfmin * pfminneg) / (prodpfmin * prevminneg + (1 - prev))).reshape(-1,)))
            pfplus += ((-1) ** len(myset)) * Xi * Di
        
        # Update bitstring v
        ready = True
        for k in range(m):
            v[k] += 1
            if v[k] <= 1:
                ready = False
                break
            v[k] = 0 if k == 0 else v[k - 1]

    pfpluspd = pfplus * prevend
    posterior = pfpluspd[1:] / pfpluspd[0]
    
    return posterior, pfpluspd, pfplus


# # Quickscore gmpy2

# In[2]:


import numpy as np
import gmpy2

# Instellen op quadrupel precisie (~float128, 113 bits)
gmpy2.get_context().precision = 113  

def fun_quickscoreW2_gmpy2(prev, pfmin, pfminneg):
    """
    Heckerman's Quickscore function
   
    High precision  version.  Output als NumPy float64 array.
    
    Arguments:
    - prev: shape (n, ) all between 0 and 1
    - pfmin: shape (m,n) all between 0 and 1
    - pfminneg: shape (n,) all between 0 and 1

    - prev: Prevalences: p(d_i=1)
    - pfmin: 1 - sensitivity (for positive tests): 1 - p(fj|only d_i =1), fj in F+
    - pfminneg: Products of 1 - sensitivity for negatives: Prod(1 - p(fj|only d_i =1)), product taken over j with fj in F-
    
    Returns:
    - posterior: Array of P(d_i=1|F+,F-) ... the most important outcome ...
    - pfpluspd: p(F+,F-) and P(F+,F-|d_i=1)p(d_i=1)  ... for debugging etc. ...
    - pfplus: p(F+,F-) and P(F+,F-|d_i=1).   ... for debugging etc. ...
   
    """
    prev = np.array(prev, dtype=object).reshape(-1)  # Object dtype voor gmpy2 compatibiliteit
    pfminneg = np.array(pfminneg, dtype=object).reshape(-1)
    n = len(prev)

    if pfmin is not None:
        m, n_pfmin = pfmin.shape
        if n_pfmin != n:
            raise ValueError('inconsistent n')
    else:
        m = 0
    
    # Hoge precisie getallen in berekeningen
    pfplus = np.array([gmpy2.mpfr(0) for _ in range(n + 1)], dtype=object)
    prevend = np.array([gmpy2.mpfr(1)] + [gmpy2.mpfr(x) for x in prev], dtype=object)
    
    if pfminneg is not None:
        prevminneg = np.array([gmpy2.mpfr(x) * gmpy2.mpfr(y) for x, y in zip(prev, pfminneg)], dtype=object)
    else:
        prevminneg = np.array([gmpy2.mpfr(x) for x in prev], dtype=object)

    v = np.zeros(m, dtype=int)
    ready = False

    while not ready:
        myset = np.where(v == 1)[0]
        
        if len(myset) == 0:
            Xi = gmpy2.mpfr(1)
            for i in range(n):
                Xi *= prevminneg[i] + (1 - prev[i])
            Di = np.array([gmpy2.mpfr(1)] + [gmpy2.mpfr(x) / (prevminneg[i] + (1 - prev[i])) for i, x in enumerate(pfminneg)], dtype=object)
            pfplus += Xi * Di
        
        elif len(myset) == 1:
            row = myset[0]
            Xi = gmpy2.mpfr(1)
            for i in range(n):
                Xi *= (pfmin[row, i] * prevminneg[i] + (1 - prev[i]))
            Di = np.array([gmpy2.mpfr(1)] + [(pfmin[row, i] * pfminneg[i]) / (pfmin[row, i] * prevminneg[i] + (1 - prev[i])) for i in range(n)], dtype=object)
            pfplus += ((-1) ** len(myset)) * Xi * Di

        else:
            prodpfmin = np.prod(pfmin[myset, :], axis=0, dtype=object)
            Xi = gmpy2.mpfr(1)
            for i in range(n):
                Xi *= prodpfmin[i] * prevminneg[i] + (1 - prev[i])
            Di = np.array([gmpy2.mpfr(1)] + [(prodpfmin[i] * pfminneg[i]) / (prodpfmin[i] * prevminneg[i] + (1 - prev[i])) for i in range(n)], dtype=object)
            pfplus += ((-1) ** len(myset)) * Xi * Di

        # Update bitstring v
        ready = True
        for k in range(m):
            v[k] += 1
            if v[k] <= 1:
                ready = False
                break
            v[k] = 0 if k == 0 else v[k - 1]

    # Omgerekend naar float64 voor output
    pfpluspd = np.array(pfplus * prevend, dtype=np.float64)
    posterior = np.array(pfpluspd[1:] / pfpluspd[0], dtype=np.float64)

    return posterior, pfpluspd, np.array(pfplus, dtype=np.float64)

