# freelancer/

Only `same_patch_pairs.json` is released from the paper's hiring kit: a label-free list of scene representatives whose
1-inch crops show the same patch of fabric (SIFT + RANSAC on the crops, `annotation_tool/make_tool_examples.py`
SameSpot), read by `paper/tools/fabric_groups_v2.py` for the re-shoot audit. The kit's job-specific builders (hosting,
names, answer keys, pay) are not released; their generic replacements are in `annotation_tool/kit/`.
