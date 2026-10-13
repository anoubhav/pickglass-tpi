import os  # added by the release export (path configuration)
import json, sys
nb=json.load(open(os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "2_Segmentation_Result_classical_CV.ipynb")))
idxs=[int(x) for x in sys.argv[1].split(',')] if len(sys.argv)>1 else range(len(nb['cells']))
for i in idxs:
    c=nb['cells'][i]
    print(f"===== CELL {i} [{c['cell_type']}] exec={c.get('execution_count')} =====")
    print(''.join(c['source']))
    for o in c.get('outputs',[]):
        t=o.get('output_type')
        if t=='stream':
            txt=''.join(o['text'])
            print(f"--- out(stream {o.get('name')}) ---"); print(txt[:4000])
        elif t in ('execute_result','display_data'):
            d=o.get('data',{})
            keys=list(d.keys())
            if 'text/plain' in d:
                print(f"--- out({t}) keys={keys} ---"); print(''.join(d['text/plain'])[:4000])
            else:
                print(f"--- out({t}) keys={keys} (no text) ---")
        elif t=='error':
            print('--- ERROR ---', o.get('ename'), o.get('evalue'))
            print('\n'.join(o.get('traceback',[]))[:3000])
