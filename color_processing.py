
# %%
import numpy as np
import cv2
import glob
# %%
def preocess_per_cell( tg='pixel'):
    assert tg in ['pixel', 'image']
    def decorator(fn):
        def wrap_image(image: np.ndarray):
            im_proc = np.zeros_like(image)
            h, w = im_proc.shape[:2]
            for i in range(h):
                for j in range(w):
                    im_proc[i, j] = fn(image[i, j])
            return im_proc
        def  wrap_cell(cell: np.ndarray):
            return (fn(cell))
        if tg=='pixel':
            return wrap_cell
        else:
            return wrap_image

    return decorator


# @preocess_per_cell(tg='pixel')
@preocess_per_cell(tg='image')
def rgb2ycbcr(rgb: np.ndarray):
    rgb = rgb.copy().astype(np.float)
    if rgb.shape[0] < 3:
        rgb = rgb.T
    ycbcr = np.array([16, 128, 128]).T +\
            1/255*np.matmul(np.array([[65.481, 128.553, 24.966],
                            [37.797, 74.203, 112.000],
                            [112.000, 93.786, 18.214]]), rgb)
    return ycbcr.astype(np.uint8)

def rgb2hsv(rgb: np.ndarray):
    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv
    return hsv
# %%
if __name__ == "__main__":
    im_list = [cv2.imread(f) for f in glob.glob('./*.jpeg')]
    print(f"{len(im_list)} images loaded!")
    im = im_list[0]
    im = cv2.resize(im, (300, int(im.shape[0]*(300/im.shape[1]))))
# %%
    cv2.imshow('Origin', im)

    res_ycrcb = rgb2ycbcr(im)
    cv2_ycrcb = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)

    res_hsv = rgb2hsv(im)
    cv2_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    cv2.imshow('My ycrcb', res_ycrcb)
    cv2.imshow('cv2 ycrcb', cv2_ycrcb)
    cv2.imshow('My hsv', res_hsv)
    cv2.imshow('cv2 hsv', cv2_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
