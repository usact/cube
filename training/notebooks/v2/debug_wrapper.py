import runpy

print("=== RUNNING train_cubediff with DEBUGGING ===")

try:
    runpy.run_path("train_cubediff.py", run_name="__main__")
except Exception as e:
    print(">>> DEBUG: Exception =", e)
    import traceback
    traceback.print_exc()
