from ex3_utils import *
import matplotlib.pyplot as plt
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float64)

    """ LK Demo """
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float64), img_2.astype(np.float64), step_size=20, win_size=5)
    et = time.time()

    print("\nLK Algorithm:",
          "\nTime: {:.4f}".format(et - st),
          "\nMedian:", np.median(uv, 0),
          "\nMean:", np.mean(uv, 0))
    displayOpticalFlow("LK", img_2, pts, uv)

    """ Hierarchical LK Demo """
    st = time.time()
    uv_hierarchical = opticalFlowPyrLK(img_1.astype(np.float64), img_2.astype(np.float64), 3, 20, 5)
    et = time.time()

    print("\nHierarchical LK Algorithm:",
          "\nTime: {:.4f}".format(et - st))
    displayOpticalFlow("Hierarchical LK", img_2, pts, uv_hierarchical.reshape((-1, 2)))


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo")

    pass


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")

    pass


def displayOpticalFlow(name: str,img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    plt.title(name)
    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")

    src = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    height, width = src.shape

    dx, dy = 50, 50
    ang = np.deg2rad(20)

    trans_mat = np.array([[1, 0, dx],
                          [0, 1, dy],
                          [0, 0, 1]])
    
    rot_mat = np.array([[np.cos(ang), -np.sin(ang), 0],
                        [np.sin(ang),  np.cos(ang), 0],
                        [0,            0,           0]])
    
    rigid_mat = np.array([[np.cos(ang), -np.sin(ang), dx],
                          [np.sin(ang),  np.cos(ang), dy],
                          [0,            0,           1]])
    

    def warp(warp_mat: np.ndarray) -> np.ndarray:

        # Generate coordinate grids for all pixels in the source image
        x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))

        # Reshape coordinate grids into column vectors
        x_indices = x_indices.reshape(-1)
        y_indices = y_indices.reshape(-1)

        # Create a matrix of homogeneous coordinates [x, y, 1]
        homogeneous_coords = np.stack((x_indices, y_indices, np.ones_like(x_indices)))

        # Apply the translation to the homogeneous coordinates
        translated_coords = warp_mat @ homogeneous_coords

        # Extract the translated x and y indices
        translated_x_indices = translated_coords[0, :]
        translated_y_indices = translated_coords[1, :]

        # Reshape the translated indices to match the source image shape
        translated_x_indices = translated_x_indices.reshape(height, width)
        translated_y_indices = translated_y_indices.reshape(height, width)

        # Clip the translated indices to stay within the source image bounds
        translated_x_indices = np.clip(translated_x_indices, 0, width - 1).astype(int)
        translated_y_indices = np.clip(translated_y_indices, 0, height - 1).astype(int)

        # return the translated image by indexing from the source image
        return src[translated_y_indices, translated_x_indices]


    def display_warpping(warped_img: np.ndarray, name: str):

        f, ax = plt.subplots(1, 2)
        f.suptitle(name)
        ax[0].imshow(src)
        ax[0].set_title('origonal')
        ax[1].imshow(warped_img)
        ax[1].set_title('warrped')
        plt.show()

    
    display_warpping(warp(trans_mat), 'translation')
    display_warpping(warp(rot_mat), 'rotation')
    display_warpping(warp(rigid_mat), 'rigid')


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():

    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    #lkDemo(img_path)
    #hierarchicalkDemo(img_path)
    #compareLK(img_path)

    imageWarpingDemo(img_path)

    #pyrGaussianDemo('input/pyr_bit.jpg')
    #pyrLaplacianDemo('input/pyr_bit.jpg')
    #blendDemo()


if __name__ == '__main__':
    main()
