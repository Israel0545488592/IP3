import numpy as np
import cv2
from functools import reduce
from typing import List, Tuple


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 212403679

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size = 10, win_size = 5) -> Tuple[np.ndarray]:
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    diff = im1 - im2
    row_derv = np.convolve(im2, np.array([-1, 0, 1]).reshape((3, 1)))
    col_derv = np.convolve(im2, np.array([-1, 0, 1]))
    translation = np.zeros(2)


    def min_cross_corr(row: int, col: int) -> Tuple[float]:

        '''least square approximation of min cross corralation of window'''

        win = np.s_[row : row + win_size, col : col + win_size]
        Ix, Iy, It = row_derv[win], col_derv[win], diff[win]

        derv_mat = np.array([[(Ix * Ix).sum(), (Ix * Iy).sum()],
                             [(Ix * Iy).sum()], (Iy * Iy).sum()])
        
        # singularity check
        ev1, ev2 = sorted(np.linalg.eigvals(derv_mat))
        if ev1 < 1 or ev2 / ev1 > 100: return np.zeros(2)

        diff_vec = -np.array([(Ix * It).sum(), (Iy * It).sum()])

        return np.linalg.inv(derv_mat) @ diff_vec
    
    min_cross_corr = np.vectorize(min_cross_corr)

    # I'm clueless about this stage, this is just a place holder
    # it sould include some update to im1 each iteration
    # and i am not sure thats the meaning of step_size
    # use reduce if possable
    for itr in range(step_size): translation += sum(min_cross_corr(im1))

    return translation
        


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    gaus_pyr1, gaus_pyr2 = gaussianPyr(img1, k), gaussianPyr(img2, k)
    # maybe def auxiliray function and use rfunctools.educe


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gau_ker_2D(size: int) -> np.ndarray:

    ker_1D = cv2.getGaussianKernel(size, -1)
    return np.outer(ker_1D, ker_1D)


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    if levels < 1: return []

    return [img] + gaussianPyr(cv2.GaussianBlur(img, (5, 5), -1)[::2, ::2], levels - 1)


def expand(img: np.ndarray, shape: Tuple[int]) -> np.ndarray:

    exp_img = np.zeros(shape)
    exp_img[::2, ::2] = img
    return cv2.filter2D(exp_img, -1, gau_ker_2D(5) * 4)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaus_pyr = gaussianPyr(img, levels)

    return [gaus_pyr[ind] - expand(gaus_pyr[ind + 1], gaus_pyr[ind].shape) for ind in range(len(gaus_pyr) -1)] + [gaus_pyr[-1]]


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    return reduce(lambda small, big : big + expand(small, big.shape), reversed(lap_pyr))


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> Tuple[np.ndarray]:
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """

    lap_pyr_1, lap_pyr_2 = laplaceianReduce(img_1, levels), laplaceianReduce(img_2, levels)
    gaus_pyr = gaussianPyr(mask, levels)
    naive_blend = img_1 * mask + img_2 * (1 - mask)
    
    def blend(ind: int) -> np.ndarray: return lap_pyr_1[ind] * gaus_pyr[ind] + (1 - gaus_pyr[ind]) * lap_pyr_2[ind]

    return naive_blend, reduce(lambda img, ind : expand(img, gaus_pyr[ind].shape) + blend(ind) , range(-2, -levels - 1, -1), blend(-1))