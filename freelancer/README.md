# freelancer/

Besides this README, this folder holds one file, `same_patch_pairs.json`: a label-free list of scene representatives whose
1-inch crops show the same patch of fabric (SIFT + RANSAC on the crops, `annotation_tool/make_tool_examples.py`
SameSpot), read by `paper/tools/fabric_groups_v2.py` for the re-shoot audit. The tools that build annotation bundles are in
`annotation_tool/kit/`.
