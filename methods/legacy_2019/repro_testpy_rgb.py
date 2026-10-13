# Reproduces the counting logic of "FR-final attempt/test.py" (lines 17,29,55,69-120) headlessly,
# under a few preprocessing variants, to see which matches xlsx column C
# "predicted weft by RGB irregularity". Read-only on the source images.
import cv2, numpy as np, os
SRC = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "legacy2019")), "FR-final attempt", "images")
xlsxC = {'a1':69,'a2':91,'a3':107,'a4':73,'a5':74,'a6':81,'a7':73,'a8':87,'a9':156,'a10':46,
         'r1':36,'r2':109,'r3':81,'r4':88,'r5':99,'r6':103,'r7':70,'r8':68,'r9':60,'r10':36}
def crossings(ch):
    avg = sum(ch)/len(ch); c = 0
    for i in range(len(ch)-1):
        if ch[i] < avg and ch[i+1] > avg: c += 1
        elif ch[i] > avg and ch[i+1] < avg: c += 1
    return c
def clahe_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); p = list(cv2.split(lab))
    p[0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(p[0])
    return cv2.cvtColor(cv2.merge(p), cv2.COLOR_LAB2BGR)
def predict(img, blur=True, clahe=False):
    img = cv2.resize(img, (600, 400))
    if clahe: img = clahe_lab(img)
    if blur: img = cv2.GaussianBlur(img, (5,5), 0)
    temp = img[:, 100:105, :].astype(np.int64)  # test.py sums uint8 with Python sum -> ints
    b = [sum(temp[i,:,0])/5 for i in range(img.shape[0])]
    g = [sum(temp[i,:,1])/5 for i in range(img.shape[0])]
    r = [sum(temp[i,:,2])/5 for i in range(img.shape[0])]
    counts = [crossings(b), crossings(r), crossings(g)]
    return counts, sum(counts)//3
names = sorted(xlsxC, key=lambda x: (x[0], int(x[1:])))
variants = {'blur(current test.py)':dict(blur=True,clahe=False),'noblur':dict(blur=False,clahe=False),
            'clahe+blur':dict(blur=True,clahe=True),'clahe_noblur':dict(blur=False,clahe=True)}
res = {v:{} for v in variants}
for n in names:
    fn = os.path.join(SRC, n + ('  .jpg' if False else ('.jpg' if n!='r6' else ' .jpg')))
    img = cv2.imread(fn)
    for v,kw in variants.items():
        res[v][n] = predict(img, **kw)
print(f"{'img':5} {'xlsxC':>6} " + " ".join(f"{v:>24}" for v in variants))
for n in names:
    print(f"{n:5} {xlsxC[n]:>6} " + " ".join(f"{str(res[v][n][1])+' '+str(res[v][n][0]):>24}" for v in variants))
for v in variants:
    m = sum(res[v][n][1]==xlsxC[n] for n in names); print(v, 'exact matches:', m, '/20')
