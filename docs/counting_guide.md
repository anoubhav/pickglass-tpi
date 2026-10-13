# How to count threads per inch

**A short guide for annotators.** All pictures in this guide are synthetic (computer-made) fabrics (`img/guide/SOURCES.json` lists the generator seeds). This is the public guide that went with the job post for PickGlass, without the hiring details. The full guide that comes with the tool is [annotator_guide.md](annotator_guide.md).

## 1. The job

We have magnified photos of woven fabric. Each photo was taken through a **pick glass**, a small magnifier with a 1-inch window, so it shows exactly **one square inch** of fabric.

For each photo you count the **threads per inch** in both directions. You do this in a simple browser tool: you put **one dot on every yarn** along two guide lines, and the tool counts the dots for you. No installation is needed.

The work starts with a short screening test (two photos). You can try the tool today on three practice photos (see section 8).

![An example on a synthetic fabric: the photo, the tool and a close-up.](img/guide/post_example.jpg)

*1: the photo through the pick glass (red = the 1-inch window). 2: the window in the tool, with one dot on every yarn the two lines cross. 3: a close-up: each dot sits on one yarn.*

## 2. Warp and weft in one picture

A woven fabric has two sets of yarns that cross at right angles. The **warp** yarns run along the length of the fabric and the **weft** yarns go across it. They pass over and under each other.

In the tool you do not need to know which set is the warp. You only look at the direction:

- The **H line** (horizontal) crosses the **vertical yarns**. Put one dot on each of them.
- The **V line** (vertical) crosses the **horizontal yarns**. Put one dot on each of them.

![Warp and weft: a drawing and a synthetic photo, with one dot per yarn on each line.](img/guide/warp_weft.jpg)

## 3. The tool

![The practice tool with 11 dots on the H line.](img/guide/practice_tool.jpg)

*The practice tool after the first 11 dots on the H line. Top: your code (Annotator), the photo number, progress and Download. Middle: the photo with the two guide lines. Right: the two counts, the weave type and the other questions.*

How to count one line:

1. Type your code in **Annotator** (the same every time).
2. **Zoom in** with the mouse wheel until the yarns are well apart. Drag an empty area to move around.
3. Start at one end of the line. **Click on the line** on every yarn it crosses: one click, one dot. Click a dot to delete it. **Z** undoes.
4. Go all the way to the other end. Count a yarn at the edge only if its centre is inside the window.
5. Press **L** to switch to the other line, and do the same.
6. Check the bars on the line: a **red bar** marks a gap (a missed yarn?) and an **orange bar** marks two dots too close (a double dot?).
7. Choose the weave type, tick any photo problems, set your confidence (3 sure, 2 mostly sure, 1 rough), then press **Enter** to save and go to the next photo.

The big number under **H line** and **V line** is the threads per inch. Quick check: count the yarns in one tenth of an inch and multiply by 10. The result should be close to the big number.

## 4. One dot per yarn: common mistakes

The rule never changes: **one dot on every single yarn** the line crosses. Patterns and colours only make the yarns harder to see.

![Three common mistakes on synthetic fabrics, each next to the right way.](img/guide/mistakes.jpg)

- **Twill (diagonal stripes):** the diagonals are not yarns. Each stripe spans several yarns, so your dots sit closer together than the stripes.
- **Two-colour plain weave:** a light float and the dark float next to it are **two** yarns, not one.
- **Checks and stripes:** a colour block is a group of yarns. Ignore the colours and put a dot on every yarn.

## 5. Weave types

![Five weave types as synthetic swatches, with their weave patterns.](img/guide/weave_types.jpg)

- **Plain (key 1):** a checkerboard. Each yarn goes over one, then under one.
- **Twill (keys 2, 3, 4):** diagonal lines. Count how many yarns one float passes over: 2 is twill 2/1, 3 is twill 3/1, and an even two-over-two-under is twill 2/2.
- **Satin (key 5):** a smooth surface with long floats and no clear diagonal.
- **Other (6)** for anything else, and **Unsure (7)** if you cannot tell after a few seconds. Unsure is a good answer; a wrong guess is not.

## 6. Photo problems

Real photos are not perfect. **Never skip a photo.** Count what you can see, tick the problem and lower your confidence.

![Six photo problems shown on one synthetic fabric.](img/guide/photo_problems.jpg)

| Problem | What you see | What to do |
| --- | --- | --- |
| **Glare** | a white, shiny patch | Drag the line to a row without glare. Tick *Glare*. |
| **Blur** | soft, fuzzy yarns | Zoom in. If you can still tell the yarns apart, count normally. Tick *Blur*. |
| **Shadow** | a dark area | Press **E** for a brighter view, or move the line out of the shadow. Tick *Shadow*. |
| **Low resolution** | blocky yarns, only 2 or 3 pixels wide | Zoom in and count anyway. Tick *Low-res*, confidence 1 or 2. |
| **Skewed window** | yarns tilted against the lines | Press **A** (turn the line) or **R** (turn the image) until the line crosses the yarns at right angles. Tick *Perspective*. |
| **Partly covered** | part of the window hidden (edge of the glass, a finger) | Move the line to the part you can see. Tick *Partially occluded*. |

If the yarns on a line cannot be seen anywhere in the window, clear that line, write why in **Notes** and set confidence 1. Do not guess a number.

## 7. What to deliver

1. Your work is saved automatically in your browser. Always use the same computer and the same browser, and never a private or incognito window.
2. When you finish, click **Download** (top right). You get **two files**: `annotations_<code>_<date>.json` and `annotations_<code>_<date>.csv`.
3. Send **both files**, and tell us roughly how long the work took.

## 8. Try it: three practice photos

Run `python3 annotation_tool/server.py --demo --open` from a clone of this repository, or open a hosted practice bundle (`annotation_tool/kit/build_bundle.py --data annotation_tool/demo/data --storage-key practice --static`).

1. Open it in Google Chrome or Microsoft Edge on a laptop or desktop computer.
2. Type any name in **Annotator**.
3. Count both lines on the 3 photos. They are synthetic fabrics made for practice.
4. Then compare your counts with the answers:

| Photo | Fabric | H line (vertical yarns) | V line (horizontal yarns) | Weave |
| --- | --- | --- | --- | --- |
| 1 of 3 (#801) | plain weave, blue | **63** | **67** | plain (key 1) |
| 2 of 3 (#802) | twill, teal | **64** | **64** | twill 2/1 (key 2) |
| 3 of 3 (#803) | two-colour plain weave, gold and dark blue | **49** | **66** | plain (key 1) |

Within **1 to 2 threads** of these numbers is fine. If you got about half on photo 3, you counted light-and-dark pairs: they are two yarns. If you got about a half or a third on photo 2, you counted the diagonal stripes.

The practice photos open in blind mode with a purple badge. That only means there are no starting dots: you place every dot yourself.
