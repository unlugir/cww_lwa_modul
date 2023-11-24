import numpy as np
import word


def alpha_cuts_intervals(m, h=1):
    """
    Returns appropriate m alpha-cuts intervals for Y_LWA
    :param m: number of interval
    :param h: smallest height of words DIT2FSs
    """
    intervals = [j / (m - 1) for j in range(m) if h >= j / (m - 1)]
    return intervals


def alpha_cut_umf(y_value, codebook):
    """
    Returns alpha-cut for i-th value for each UMFs in codebook (UMFs suppose have trapezoidal type)
    :param y_value: alpha-cut interval for Y_LWA
    :param codebook: appropriate codebook
    """
    result = []
    for fou in codebook['words'].values():
        mf_type, a, b, c, d = fou['umf']
        x_alpha = ((b - a) * y_value + a, d - (d - c) * y_value)
        result.append(x_alpha)
    return result


def alpha_cut_lmf(y_value, codebook):
    """
    Returns alpha-cut for i-th value for each LMFs in codebook (LMFs suppose have trapezoidal type)
    :param y_value: alpha-cut interval for Y_LWA
    :param codebook: appropriate codebook
    """
    result = []
    for fou in codebook['words'].values():
        mf_type, a, b, c, d, h = fou['lmf']
        x_alpha = ((b - a) * y_value + a, d - (d - c) * y_value)
        result.append(x_alpha)
    return result


def y_umf(intervals, codebook, W):
    """
    Returns alpha-cuts for UMF Y_LWA
    :param intervals: alpha-cuts intervals
    :param codebook: model of words
    :param W: list of weights
    """
    umf_result = []
    for y_value in intervals:
        x_range = alpha_cut_umf(y_value, codebook)
        x_left = sum(x[0] * w for x, w in zip(x_range, W)) / sum(W)
        x_right = sum(x[1] * w for x, w in zip(x_range, W)) / sum(W)
        umf_result.append((x_left, x_right))
    return umf_result


def y_lmf(intervals, codebook, W):
    """
    Returns alpha-cuts for LMF Y_LWA
    :param intervals: alpha-cuts intervals
    :param codebook: model of words
    :param W: list of weights
    """
    lmf_result = []
    for y_value in intervals:
        x_range = alpha_cut_lmf(y_value, codebook)
        x_left = sum(x[0] * w for x, w in zip(x_range, W)) / sum(W)
        x_right = sum(x[1] * w for x, w in zip(x_range, W)) / sum(W)
        lmf_result.append((x_left, x_right))
    return lmf_result


def construct_dit2fs(x, intervals_lmf, y_lmf, intervals_umf, y_umf):
    umf = np.zeros(len(x))
    lmf = np.zeros(len(x))

    for index, y in enumerate(intervals_umf):
        x_left, x_right = y_umf[index]
        left_index = np.argmax(x >= x_left)
        right_index = np.argmax(x >= x_right)
        for i in range(left_index, right_index + 1):
            umf[i] = y

    for index, y in enumerate(intervals_lmf):
        x_left, x_right = y_lmf[index]
        left_index = np.argmax(x >= x_left)
        right_index = np.argmax(x >= x_right)

        for i in range(left_index, right_index + 1):
            lmf[i] = y

    return word.Word(None, x, lmf, umf, base_on='custom')