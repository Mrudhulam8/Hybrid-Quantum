"""Runs pytest-style tests when pytest isn't installed (supports tmp_path, monkeypatch)."""
import importlib, inspect, sys, tempfile, pathlib, traceback, unittest

class MP:
    def __init__(self): self.undo=[]
    def setitem(self, d, k, v): self.undo.append((d,k,d.get(k))); d[k]=v
    def restore(self):
        for d,k,v in reversed(self.undo): d[k]=v

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
fails=0
for modname in sys.argv[1:]:
    mod=importlib.import_module(modname)
    for name,fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"): continue
        kw={}; mp=MP()
        if "tmp_path" in inspect.signature(fn).parameters: kw["tmp_path"]=pathlib.Path(tempfile.mkdtemp())
        if "monkeypatch" in inspect.signature(fn).parameters: kw["monkeypatch"]=mp
        try: fn(**kw); print("PASS", name)
        except unittest.SkipTest as e: print("SKIP", name, "-", e)
        except Exception: fails+=1; print("FAIL", name); traceback.print_exc()
        finally: mp.restore()
print("failures:", fails); sys.exit(1 if fails else 0)
