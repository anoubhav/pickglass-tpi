# Headless re-run of TQA/doc3.py logic (lines 8-88) on TQA/images, writing nothing to the source tree;
# compares with TQA/outputs/f*.jpg.jpg and applies TQA/count.py logic (lines 5-15).
import cv2, numpy as np, os
from scipy import ndimage as inter
T = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "legacy2019")), "TQA")
def count_py(img):
    h,w = img.shape[:2]; warp1 = 0
    for y in range(0,1):
        for x in range(0,w):
            if img[y,x] - img[y,x-1] == 255: warp1 += 1   # uint8 arithmetic, as in count.py
    return warp1
def transitions_all_rows(b):
    bb = (b > 127).astype(np.int8)
    return np.abs(np.diff(bb, axis=1)).sum(axis=1), np.abs(np.diff(bb, axis=0)).sum(axis=0)
def doc3(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(imgray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)
    thresh = cv2.blur(thresh, (3,3))
    k = 40; h, w = thresh.shape
    filt = thresh[int(h/2-k):int(h/2+k), int(w/2-k):int(w/2+k)]
    filt = 2*filt-1; filt = filt*255/(4*k*k); filti = filt*-1
    thresh = thresh*2-np.ones(np.shape(thresh))
    f = cv2.filter2D(thresh, cv2.CV_8UC3, filt); fi = cv2.filter2D(thresh, cv2.CV_8UC3, filti)
    f = cv2.subtract(f, fi)
    def find_score(im, angle):
        data = inter.rotate(im, angle, reshape=False, order=0); hist = np.sum(data, axis=1)
        return hist, np.sum((hist[1:] - hist[:-1]) ** 2)
    angles = range(-10, 10, 1); scores = [find_score(f, a)[1] for a in angles]
    best = angles[scores.index(max(scores))]
    f = inter.rotate(f, best, reshape=False, order=0)
    f2 = f - cv2.blur(f, (51,51)) + 125
    f = cv2.dilate(f, np.ones((3,3),np.uint8), iterations=1)
    Z = np.float32(f2.reshape((-1,1)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cv2.setRNGSeed(0)
    ret, label, center = cv2.kmeans(Z, 3, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
    for i in range(3):
        if np.max(center)==center[i]: a=i
        elif np.min(center)==center[i]: c=i
        else: b=i
    if center[b] < 128: b = a
    f = cv2.adaptiveThreshold(f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, float(-np.abs(center[c]-center[b])*1.3/2))
    return f[int(h/2-k):int(h/2+k), int(w/2-k):int(w/2+k)], best, center.ravel()
print("count.py on TQA/final.jpg:", count_py(cv2.imread(T+"/final.jpg",0)))
names = sorted([n for n in os.listdir(T+"/images") if n.endswith('.jpg')], key=lambda s:int(s[1:-4]))
for n in names:
    out, ang, cen = doc3(cv2.imread(T+"/images/"+n))
    saved = cv2.imread(T+"/outputs/"+n+".jpg", 0)
    agree = ((out>127)==(saved>127)).mean()
    r,c = transitions_all_rows(saved)
    print(f"{n:8} angle={ang:4d} centers={np.round(cen,1)} pix_agree_vs_saved={agree:.3f} count.py(saved)={count_py(saved)} "
          f"rowTrans max/med={r.max()}/{int(np.median(r))} colTrans max/med={c.max()}/{int(np.median(c))}")
