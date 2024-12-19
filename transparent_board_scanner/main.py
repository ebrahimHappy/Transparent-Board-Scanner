
# %%
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


# %%
images = [
    cv.imread(f'../samples/hello-world/image{i}.jpg')[...,::-1]
    for i in range(1, 7)
]


# %%



class ImageAligner:
    def __init__(self):
        self.sift = cv.SIFT_create()

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

        self.good_match_ratio = 0.85

        self.min_match_count = 20

    def _calculate_keypoints_descriptors(self, image):
        grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        grayscale = cv.medianBlur(grayscale, 3)
        grayscale = cv.morphologyEx(grayscale, cv.MORPH_BLACKHAT, np.ones((11,11), 'uint8')) + 127
        keypoints, descriptors = self.sift.detectAndCompute(grayscale, None)
        return keypoints, descriptors

    def _filter_good_matches(self, matches):
        return [m for m, n in matches if m.distance / n.distance < self.good_match_ratio]

    def _find_homography_transform(self, matches, keypoints1, keypoints2):
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        transform, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if mask.sum() < self.min_match_count:
            return None
        return transform

    def set_reference_image(self, image):
        keypoints, descriptors = self._calculate_keypoints_descriptors(image)
        self.reference_keypoints = keypoints
        self.reference_descriptors = descriptors
        self.reference_shape = image.shape[:2]
        return self

    def align_image(self, image):
        keypoints, descriptors = self._calculate_keypoints_descriptors(image)
        matches = self.flann.knnMatch(descriptors, self.reference_descriptors, k=2)
        good_matches = self._filter_good_matches(matches)
        if len(good_matches) < self.min_match_count:
            return None
        transform = self._find_homography_transform(good_matches, keypoints, self.reference_keypoints)
        if transform is None:
            return None
        aligned = cv.warpPerspective(image, transform, self.reference_shape[::-1])
        return aligned

# %%

aligner = ImageAligner().set_reference_image(images[0])
aligned = [images[0]] + [aligner.align_image(img) for img in images[1:]]
aligned = [img for img in aligned if img is not None]


# %%

def highpass(img, sigma):
    return img - cv.GaussianBlur(img, (0,0), sigma) + 127


# %%
agg = []
for i, img in enumerate(aligned):
    img = cv.medianBlur(img, 3)
    hp = highpass(img, 10)
    cv.imwrite(f'../outputs/hp{i}.jpg', hp)
    hp = np.clip(hp, 0, 127)
    agg.append(hp)
result = (np.median(agg, axis=0) * 2).astype('uint8')

plt.imshow(result)

cv.imwrite('../outputs/result.jpg', result[...,::-1])

# %%
for i in range(len(aligned)):
    cv.imwrite(f'../outputs/align{i}.jpg', aligned[i][...,::-1])

# %%
winSize = (48,48)
blockSize = (48,48)
blockStride = (16,16)
cellSize = (16,16)
nbins = 9
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

fvs = []
for img in aligned:
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.GaussianBlur(img, (0,0), 3)
    fv = hog.compute(img) 
    fvs.append(fv.reshape(-1, 81))
fvs = np.array(fvs)

# %%
similarities = (fvs * fvs[:, np.newaxis, ...]).sum(axis=-1)
similarities[np.eye(5, dtype=bool)] = 0
weights = similarities.max(axis=0).reshape(-1, (4000 - 32) // 16 , (3000 - 32) // 16)

# %%
# for i, img in enumerate(aligned):
#     img = cv.medianBlur(img, 3)
#     hp = highpass(img, 10)
#     mask = cv.resize(weights[i], (x.shape[1] * 16, x.shape[0]*16))
#     hp = hp[:, :-8][16:-16, 16:-16]
#     plt.imshow()

# %%
xs = []
for i, a in enumerate(fvs):
    for b in fvs[i+1:]:
        x = (a * b).sum(axis=1).reshape((4000 - 32) // 16 , (3000 - 32) // 16)
        xs.append(x)
        # plt.imshow(x)
        # plt.show()
x = np.max(xs, axis=0)
# %%
mask = cv.resize(x, (x.shape[1] * 16, x.shape[0]*16))
# plt.imshow(images[0][:-8, :-8][16:-16, 16:-16])
# plt.imshow(mask, alpha=0.5)
# %%
masked_result = 255-((255-result[:, :-8][16:-16, 16:-16]) * (mask[...,np.newaxis] > .9))
cv.imwrite('../outputs/masked.jpg', masked_result)

# %%
