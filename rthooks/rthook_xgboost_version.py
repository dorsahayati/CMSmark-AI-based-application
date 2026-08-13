"""
PyInstaller runtime hook — belt-and-suspenders fallback for xgboost VERSION.

This runs before any user code. If the spec-level datas fix already placed
xgboost/VERSION in sys._MEIPASS this is a no-op. Otherwise it creates the
file so that xgboost.__init__ can read it (via open() or importlib.resources)
without raising FileNotFoundError and blocking model pickle loading.
"""
import sys
import os

if getattr(sys, 'frozen', False):
    _xgb_dir  = os.path.join(sys._MEIPASS, 'xgboost')
    _ver_file = os.path.join(_xgb_dir, 'VERSION')
    if not os.path.exists(_ver_file):
        os.makedirs(_xgb_dir, exist_ok=True)
        # Read real version from bundled DLL so it matches the C library check
        import ctypes
        _ver = '2.0.0'
        try:
            _dll = os.path.join(_xgb_dir, 'lib', 'xgboost.dll')
            _lib = ctypes.CDLL(_dll)
            _maj, _min, _pat = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
            _lib.XGBoostVersion(ctypes.byref(_maj), ctypes.byref(_min), ctypes.byref(_pat))
            _ver = f'{_maj.value}.{_min.value}.{_pat.value}'
        except Exception:
            pass
        with open(_ver_file, 'w', encoding='ascii') as _f:
            _f.write(_ver)
