import numpy as np
import findiff
import pandas as pd


def calculate_findiff_coefs(offsets, deriv):
    """
    Calculate Finite Difference Coefficients
    
    Parameters
    ----------
    offsets : array_like
    deriv : int
    
    Returns
    -------
    coefs : ndarray
    
    Examples
    --------
    >>> calculate_findiff_coefs([-2, -1, 0, 1, 2], 4) 
    array([ 1., -4.,  6., -4.,  1.])
    
    >>> calculate_findiff_coefs([-2, -1, 0, 1, 2, 1.99], 3)
    array([  -0.1867,   -0.6722,    3.7688,   -6.0505, -124.5   ,  127.6406])
    
    References
    ----------
    https://web.media.mit.edu/~crtaylor/calculator.html
    https://github.com/maroba/findiff/blob/e8ca33707e3e25d76bf0f93b2391e466209287b1/findiff/coefs.py
    """
    
    if len(offsets) < deriv + 1:
        raise ValueError("Length of offsets should be larger than derivative order plus 1.")
    if len(offsets) != len(set(offsets)):
        # Note that this program could not handle pathological cases, only make simple check instead.
        raise ValueError("Possibly exactly same offset value is given. Please check `offsets'.")
    
    offsets = np.asarray(offsets)
    matrix = np.array([offsets**n for n in range(len(offsets))])
    rhs = np.zeros(len(offsets))
    rhs[deriv] = np.math.factorial(deriv)
    
    return np.linalg.solve(matrix, rhs)


def calculate_GRR_trig(offsets_half, deriv, fx_pos, fx_neg, f0=None):
    """
    Calculate (Generalized) Rutishauser–Romberg Triangle
    
    Parameters
    ----------
    offsets_half : array_like
        Positive half part of offsets.  For example, if all offsets are [-2, -1, 0, 1, 2],
            then `offsets_half' should be [1, 2].
        Must be a geometric sequence. This function does not make double check on this.
        Do not contain zero in this array.
    deriv : int
    fx_pos : array_like
        Value list of f(offsets_half). Dimension should be the same to `offsets_half'.
    fx_neg : array_like
        Value list of f(-offsets_half). Dimension should be the same to `offsets_half'.
    f0 : float or None
        Value of f(0). May leave as None if derivative order is odd number.
    
    Returns
    -------
    grr_trig : ndarray
        (Generalized) Rutishauser–Romberg triangle.
    """
    
    if deriv % 2 == 0 and f0 is None:
        raise ValueError("`f0' should be provided if derivative order is even number.")
    else:
        f0 = 0 if f0 is None else f0
    if len(offsets_half) < 2:
        raise ValueError("length of `offsets_half' must >= 2 in order to calculate ratio.")

    comp_len = (deriv + 1) // 2
    mat_size = len(offsets_half) - comp_len
    grr_trig = np.zeros((mat_size, mat_size))
    ratio = offsets_half[-1] / offsets_half[-2]
    
    for r in range(mat_size):
        i_end = r + comp_len
        offsets = np.concatenate([offsets_half[r:i_end], -offsets_half[r:i_end], [0]])
        coef_list = calculate_findiff_coefs(offsets, deriv)
        val_list = np.concatenate([fx_pos[r:i_end], fx_neg[r:i_end], [f0]])
        grr_trig[r, 0] = (coef_list * val_list).sum()
    for c in range(1, mat_size):
        for r in range(mat_size-c):
            grr_trig[r, c] = (ratio**(2*c) * grr_trig[r, c-1] - grr_trig[r+1, c-1]) / (ratio**(2*c) - 1)    
            
    for r in range(mat_size):
        for c in range(mat_size-r, mat_size):
            grr_trig[r, c] = np.nan
    return grr_trig


def calculate_GRR_trig_with_f(offsets_half, deriv, f):
    """
    Calculate (Generalized) Rutishauser–Romberg Triangle with Given Function
    
    Parameters
    ----------
    offsets_half : array_like
    deriv : int
    f : function
        Should be able to handle both array_like and float input.
    
    Returns
    -------
    grr_trig : ndarray
        (Generalized) Rutishauser–Romberg triangle.
    """
    
    fx_pos = [f(o) for o in offsets_half]
    fx_neg = [f(-o) for o in offsets_half]
    f0 = f(0)
    return calculate_GRR_trig(offsets_half, deriv, fx_pos, fx_neg, f0)


def check_grr_trig_converge(grr_trig):
    """
    Convergence Check Matrix of (Generalized) Rutishauser–Romberg Triangle
    
    Parameters
    ----------
    grr_trig : ndarray
        (Generalized) Rutishauser–Romberg triangle.
    
    Return
    ------
    mat_chk : ndarray
    """
    
    n = len(grr_trig)
    mat_chk = np.zeros((n, n))
    for r in range(0, n-1):
        for c in range(1, n-r-1):
            mat_chk[r, c] = np.abs(grr_trig[r, c] - grr_trig[r+1, c]) + np.abs(grr_trig[r, c] - grr_trig[r, c-1])
    for r in range(n):
        mat_chk[r, 0] = np.nan
    for r in range(0, n):
        for c in range(n-r-1, n):
            mat_chk[r, c] = np.nan
    return mat_chk


def output_pd_grr_trig(offsets_half, grr_trig, tolerance=3):
    """
    Pandas Presentation of (Generalized) Rutishauser–Romberg Triangle
    
    Parameters
    ----------
    offsets_half : array_like
    grr_trig : ndarray
        (Generalized) Rutishauser–Romberg triangle.
    tolerance : int
        Number of minimum difference cells in convergence check matrix.
    
    Return
    ------
    df : pandas.io.formats.style.Styler
        Pandas show of GRR triangle.
    df_check : pandas.io.formats.style.Styler
        Pandas show of convergence check matrix of GRR triangle.
    """
    n = len(grr_trig)
    df = pd.DataFrame(grr_trig, columns=range(n), index=offsets_half[:n])
    df_check = pd.DataFrame(check_grr_trig_converge(grr_trig), columns=range(n), index=offsets_half[:n])
    df.replace(np.nan, "", regex=True, inplace=True)
    df_check.replace(np.nan, "", regex=True, inplace=True)

    t = check_grr_trig_converge(grr_trig).flatten()
    t = t.argsort()[:3]
    t = np.array([t // n, t % n]).T
    
    def highlight_cells(x):
        df = x.copy()
        df.loc[:,:] = '' 
        for r, c in t:
            df.iloc[r, c] = "background-color: lightgreen"
            df.iloc[r, c-1] = "background-color: lightgreen"
            df.iloc[r+1, c] = "background-color: lightgreen"
        return df

    df = df.style.apply(highlight_cells, axis=None)
    df_check = df_check.style.apply(highlight_cells, axis=None)
    return df, df_check
