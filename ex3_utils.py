import numpy as np
import cv2
from functools import reduce
from itertools import product
from copy import copy
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
    diff = im2 - im1
    row_derv = cv2.filter2D(im2, -1, np.array([[-1, 0, 1]]))
    col_derv = cv2.filter2D(im2, -1, np.array([[-1, 0, 1]]).transpose())

    def min_cross_corr(row: int, col: int) -> Tuple[float]:

        '''least square approximation of min cross corralation of a window'''

        win = np.s_[row : row + win_size, col : col + win_size]
        Ix, Iy, It = row_derv[win], col_derv[win], diff[win]

        mcc_mat = np.array([[(Ix * Ix).sum(), (Ix * Iy).sum()],
                             [(Ix * Iy).sum(), (Iy * Iy).sum()]])
        
        # singularity check
        ev1, ev2 = sorted(np.linalg.eigvals(mcc_mat))
        if ev1 < 1 or ev2 / ev1 > 100: return np.zeros(2)

        diff_vec = -np.array([(Ix * It).sum(), (Iy * It).sum()])

        return np.linalg.inv(mcc_mat) @ diff_vec


    indxs = product(range(0, im1.shape[0], step_size), range(0, im1.shape[1], step_size))

    return np.array([(col, row) for row, col in copy(indxs)]), np.array([min_cross_corr(row, col) for row, col in copy(indxs)])
    
        


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

    def flow(ind: int):

        pts, motion = opticalFlow(gaus_pyr1[ind], gaus_pyr2[ind], stepSize, winSize)
        return motion.reshape(gaus_pyr1[ind].shape[0] // stepSize, gaus_pyr1[ind].shape[1] // stepSize, 2)
    
    def add_flows(small: np.ndarray, big: np.ndarray):

        exp = np.zeros(big.shape)
        exp[::2, ::2] = 2 * small
        return exp + big
        
    return reduce(lambda motion, ind : add_flows(motion, flow(ind)), range(-2, -k -1, -1), flow(-1))



# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findAffineLK(im1: np.ndarray, im2: np.ndarray, motion2matrix, min_err: float = 0.01) -> np.ndarray:

    prominent_fetures = cv2.goodFeaturesToTrack(im1, maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
    directions = cv2.calcOpticalFlowPyrLK(im1, im2, prominent_fetures, maxLevel = 5) - prominent_fetures

    ans = np.zeros((2, 3))
    error = np.inf
    wrp_im = copy(im1)

    for quiver in directions:

        if error < min_err: break

        wrp_mat = motion2matrix(quiver[0, 0], quiver[0, 1])
        wrp_im = cv2.warpAffine(im1, wrp_mat, im1.shape[::-1])
        mse = np.square(im2 - wrp_im).mean()

        if mse < error: error, ans = mse, wrp_mat

    return ans
            


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    return findAffineLK(im1, im2, lambda dx, dy : np.array([[1, 0, dx], [0, 1, dy]], dtype = np.float32))


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    def motion2rot_mat(dx: float, dy: float) -> np.ndarray:

        ang = np.arctan(dy / dx) if dx != 0 else 0
        return np.array([[np.cos(ang), -np.sin(ang), dx], [np.sin(ang),  np.cos(ang), dy]], dtype = np.float32)
    
    # finding rotation first and then translation (translation ain't linear and thus the order matters)
    rot_mat = findAffineLK(im1, im2, motion2rot_mat)
    rot_img = cv2.warpAffine(im1, rot_mat, im1.shape[::-1])
    trans_mat = findTranslationLK(rot_img, im2)

    return rot_mat @ trans_mat


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