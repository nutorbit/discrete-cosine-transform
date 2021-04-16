import numpy as np


def get_quantization_matrix(quality: int = 100) -> np.ndarray:
    """
    Get quantization matrix denpends on quality.

    Args:
        quality: quality of quantization

    Returns:
        quantiazation matrix
    """

    # quantization matrix
    q = [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]]

    q = np.array(q)

    if quality == 100:
        return q
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    q = np.floor((q * scale + 50) / 100)
    return q
