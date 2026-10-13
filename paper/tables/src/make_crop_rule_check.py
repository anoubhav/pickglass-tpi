import sys, os, cv2, numpy as np
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, W + "/vlm"); sys.path.insert(0, W + "/methods")
import data_vlm as D, legacy as L
names = ["a1","a2","a4","a5","a6","a8","a10","r2","r3","r4","r6","r7","r8"]
tiles = []
for n in names:
    img = L.load_legacy(n); h, w = img.shape[:2]
    x0,y0,x1,y1 = D.detect_legacy_window(img)
    c, _ = L.detect_window(img)
    vis = img.copy()
    t = max(3, w // 300)
    cv2.rectangle(vis, (x0,y0), (x1-1,y1-1), (0,255,0), t)
    cv2.polylines(vis, [c.astype(np.int32).reshape(-1,1,2)], True, (0,0,255), t)
    cv2.rectangle(vis, (0,0), (w-1,h-1), (255,0,0), t)
    v = cv2.resize(vis, (520, 520))
    cv2.putText(v, n, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 3)
    tiles.append(v)
while len(tiles) % 5: tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i+5]) for i in range(0, len(tiles), 5)])
cv2.imwrite(sys.argv[1], grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
# zoomed corners: top-left 300x300 of 4 worst-trim images
z = []
for n in ["r2","r8","r9","r4","a4","r6"]:
    img = L.load_legacy(n); h, w = img.shape[:2]
    x0,y0,x1,y1 = D.detect_legacy_window(img)
    c, _ = L.detect_window(img)
    vis = img.copy()
    cv2.rectangle(vis, (x0,y0), (x1-1,y1-1), (0,255,0), 2)
    cv2.polylines(vis, [c.astype(np.int32).reshape(-1,1,2)], True, (0,0,255), 2)
    for (yy, xx) in [(0,0),(0,w-300),(h-300,0),(h-300,w-300)]:
        p = cv2.resize(vis[yy:yy+300, xx:xx+300], (300,300))
        cv2.putText(p, n, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        z.append(p)
grid2 = np.vstack([np.hstack(z[i:i+4]) for i in range(0, len(z), 4)])
cv2.imwrite(sys.argv[2], grid2, [cv2.IMWRITE_JPEG_QUALITY, 85])
