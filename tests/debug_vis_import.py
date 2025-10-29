'''
Author: zy
Date: 2025-10-24 21:10:35
LastEditTime: 2025-10-24 21:14:58
LastEditors: zy
Description: 
FilePath: /haitianbei/tests/debug_vis_import.py

'''
import os, sys, traceback
print("PY", sys.version)
try:
    import matplotlib
    print("matplotlib:", getattr(matplotlib, "__version__", "?"))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    import networkx as nx
    print("networkx:", getattr(nx, "__version__", "?"))
    import PIL
    print("Pillow:", getattr(PIL, "__version__", "?"))
    os.makedirs("output", exist_ok=True)
    fig = plt.figure()
    plt.plot([1,2,3],[1,4,9])
    out = os.path.join("output","_debug_plot.png")
    fig.savefig(out, dpi=50)
    print("Saved:", out, os.path.exists(out), os.path.getsize(out) if os.path.exists(out) else 0)
except Exception as e:
    print("[ERROR]", e)
    traceback.print_exc()
