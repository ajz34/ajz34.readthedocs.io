import numpy as np
import scipy
from sklearn import metrics
from scipy.spatial.distance import cdist


def ks_from_hxhc(spectra, test_size=0.25, metric='euclidean', *args, **kwargs):
    """
    Obtained from
    https://hxhc.github.io/post/kennardstone-spxy/
    """

    if test_size < 1:
        train_size = round(spectra.shape[0] * (1 - test_size))
    else:
        train_size = spectra.shape[0] - round(test_size)

    if train_size > 2:
        distance = cdist(spectra, spectra, metric=metric, *args, **kwargs)
        select_pts, remaining_pts = max_min_distance_split(distance, train_size)
    else:
        raise ValueError("train sample size should be at least 2")

    return select_pts, remaining_pts


def max_min_distance_split(distance, train_size):
    """
    Obtained from
    https://hxhc.github.io/post/kennardstone-spxy/
    """
    select_pts = []
    remaining_pts = [x for x in range(distance.shape[0])]

    # first select 2 farthest points
    first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
    select_pts.append(first_2pts[0])
    select_pts.append(first_2pts[1])

    # remove the first 2 points from the remaining list
    remaining_pts.remove(first_2pts[0])
    remaining_pts.remove(first_2pts[1])

    for i in range(train_size - 2):
        # find the maximum minimum distance
        select_distance = distance[select_pts, :]
        min_distance = select_distance[:, remaining_pts]
        min_distance = np.min(min_distance, axis=0)
        max_min_distance = np.max(min_distance)

        # select the first point (in case that several distances are the same, choose the first one)
        points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()
        for point in points:
            if point in select_pts:
                pass
            else:
                select_pts.append(point)
                remaining_pts.remove(point)
                break
    return select_pts, remaining_pts


def ks_from_karoka(X, k):
    """
    Modified from
    https://github.com/karoka/Kennard-Stone-Algorithm/blob/master/kenStone.py
    """
    n = len(X) # number of samples
    assert n >= 2 and n >= k and k >= 2, "Error: number of rows must >= 2, k must >= 2 and k must > number of rows"
    # pair-wise distance matrix
    dist = metrics.pairwise_distances(X, metric='euclidean', n_jobs=-1)

    # get the first two samples
    i0, i1 = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    selected = set([i0, i1])
    selected_list = [i0, i1]
    k -= 2
    # iterate find the rest
    minj = i0
    while k > 0 and len(selected) < n:
        mindist = 0.0
        for j in range(n):
            if j not in selected:
                mindistj = min([dist[j][i] for i in selected])
                if mindistj > mindist:
                    minj = j
                    mindist = mindistj
        selected.add(minj)
        selected_list.append(minj)
        k -= 1
    # return selected samples
    return selected_list


def ks_from_XiaqiongFan(X, Num):
    """
    Modified from
    https://github.com/XiaqiongFan/PC-CCA/blob/master/PC-CCA.py
    """
    nrow = X.shape[0]
    CalInd = np.zeros((Num), dtype=int)-1
    vAll = np.arange(0, nrow)
    D = np.zeros((nrow, nrow))
    for i in range(nrow-1):
        for j in range(i+1, nrow):
            D[i, j] = np.linalg.norm(X[i, :]-X[j, :])
    ind = np.where(D == D.max())
    CalInd[0] = ind[1]
    CalInd[1] = ind[0]
    for i in range(2, Num):
        vNotSelected = np.array(list(set(vAll)-set(CalInd)))
        vMinDistance = np.zeros(nrow-i)
        for j in range(nrow-i):
            nIndexNotSelected = vNotSelected[j]
            vDistanceNew = np.zeros((i))
            for k in range(i):
                nIndexSelected = CalInd[k]
                if nIndexSelected <= nIndexNotSelected:
                    vDistanceNew[k] = D[nIndexSelected,nIndexNotSelected]
                else:
                    vDistanceNew[k] = D[nIndexNotSelected, nIndexSelected]
            vMinDistance[j] = vDistanceNew.min()
        nIndexvMinDistance = np.where(vMinDistance == vMinDistance.max())
        CalInd[i] = vNotSelected[nIndexvMinDistance]
    ValInd = np.array(list(set(vAll)-set(CalInd)))
    return CalInd, ValInd
