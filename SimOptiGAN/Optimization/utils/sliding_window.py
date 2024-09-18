import itertools
import math

import numpy as np


def generateSlices(im_shape, maxWindowSize, overlapPercent):
    """
    Generates a set of sliding windows for a dataset with the specified dimensions and order.
    """

    # If the input data is smaller than the specified window size,
    # clip the window size to the input size on both dimensions
    windowSize = [min(maxWindowSize[ind], im_shape[ind]) for ind in range(len(im_shape))]

    # Compute the window overlap and step size
    if not isinstance(overlapPercent, (list, tuple)):
        overlapPercent = (overlapPercent, overlapPercent, overlapPercent)

    windowOverlap = [int(math.floor(windowSize[ind] * overlapPercent[ind])) for ind in range(len(im_shape))]
    stepSize = [windowSize[ind] - windowOverlap[ind] for ind in range(len(im_shape))]

    # Determine how many windows we will need in order to cover the input data
    last = [im_shape[ind] - windowSize[ind] for ind in range(len(im_shape))]
    offsets = [list(range(0, last[ind]+1, stepSize[ind])) for ind in range(len(im_shape))]

    # Unless the input data dimensions are exact multiples of the step size,
    # we will need one additional row and column of windows to get 100% coverage
    for ind in range(len(im_shape)):
        if len(offsets[ind]) == 0 or offsets[ind][-1] != last[ind]:
            offsets[ind].append(last[ind])

    # Generate the list of windows
    windows = []
    for offset in itertools.product(*offsets):
        windows.append(tuple(
            [slice(offset[ind], offset[ind]+windowSize[ind]) for ind in range(len(im_shape))]
        ))
    overlap = tuple([windowOverlap for ind in range(len(im_shape))])
    return windows, overlap


def __get_slices_halfcut(sl_in, p_size, cut_point, im_shape):
    '''Calculate the slice indices for the halfcut variant'''
    sl_out = list(sl_in)
    sl_res = [slice(0, val) for val in p_size]

    for ind in range(len(sl_in)):
        if sl_in[ind].start != 0: # if not at boarder start
            sl_out[ind] = slice(sl_out[ind].start + cut_point[ind,0], sl_out[ind].stop)
            sl_res[ind] = slice(sl_res[ind].start + cut_point[ind,0], sl_res[ind].stop)
        if sl_in[ind].stop != im_shape[ind]: # X, if not at boarder end
            sl_out[ind] = slice(sl_out[ind].start, sl_out[ind].stop-cut_point[ind,1])
            sl_res[ind] = slice(sl_res[ind].start, sl_res[ind].stop-cut_point[ind,1])

    return tuple(sl_out), tuple(sl_res)


def generateSlices_half_cut(im_shape, p_size, overlap):
    if not isinstance(overlap, (list, tuple)):
        overlap = (overlap,)*len(im_shape)
    slices_in, _ = generateSlices(im_shape, p_size, overlap)

    cut_point = np.asarray([
        [np.floor(np.floor(p_size[ind]*overlap[ind])/2), np.ceil(np.floor(p_size[ind]*overlap[ind])/2)] for ind in range(len(p_size))
        ]).astype(np.uint16)

    slices_out, slices_res = [], []
    for sl_in in slices_in:
        sl_out, sl_res = __get_slices_halfcut(sl_in, p_size, cut_point, im_shape)
        slices_out.append(sl_out)
        slices_res.append(sl_res)
    
    return slices_in, slices_out, slices_res