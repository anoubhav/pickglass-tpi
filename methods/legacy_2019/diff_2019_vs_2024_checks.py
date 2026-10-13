"""Quantify two pre-k-means differences between 2019 filterkmhigh(sari).py and notebook cells[26]:
(1) bilateral order: 2019 bilateralFilter(thresh{0,1}) then *255  vs  2024 *255 then bilateralFilter
(2) combination: 2019 f3/2 + vis/2 (float)  vs  2024 (f3 + vis)/2 with uint8 operands (wraps >255)."""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seg_final as S
for p in [os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "final_image_set", "63.jpeg"),
          os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "final_image_set", "15.jpeg"),
          os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "final_image_set", "89.jpeg")]:
    _, _, gray, _ = S.preprocess_image(p)
    th = S.apply_thresholding(gray)
    f3 = S.apply_custom_filter(th)
    # 2019 order (filterkmhigh.py L65-66)
    v19 = 255 * cv2.bilateralFilter(th, 3, 5, 5)
    # 2024 order (cells[26] apply_bilateral_filtering)
    v24 = cv2.bilateralFilter(th * 255, 3, 5, 5)
    print(os.path.basename(p), "thresh dtype", th.dtype, "| bilateral: pixels differing 2019 vs 2024 = %.2f%%" % (100 * (v19 != v24).mean()),
          "| 2024 bilateral == input*255: %s" % np.array_equal(v24, th * 255))
    k = np.ones((7, 7), np.uint8)
    b19, b24 = cv2.dilate(v19, k), cv2.dilate(v24, k)
    for _ in range(6):
        b19 = cv2.blur(b19, (25, 25)); b24 = cv2.blur(b24, (25, 25))
    f3b = cv2.blur(f3, (30, 30))
    comb19 = f3b / 2 + b24 / 2
    comb24 = S.combine_custom_and_bilateral(f3, b24)
    print("   dtypes f3b %s bil %s | pixels where f3b+bil>255 (uint8 wrap in 2024): %.2f%% | max |comb24-comb19| = %.1f" % (
        f3b.dtype, b24.dtype, 100 * ((f3b.astype(int) + b24.astype(int)) > 255).mean(), np.abs(comb24 - comb19).max()))
