# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs, collect_data_files

# SPECPATH is the directory containing this spec file — works on any machine/OS
src_dir = os.path.join(SPECPATH, 'src')
icon_dir = os.path.join(SPECPATH, 'icon')

hiddenimports = []
hiddenimports += collect_submodules('sklearn')
hiddenimports += ['xgboost', 'xgboost.sklearn', 'xgboost.core', 'xgboost.training']

# Bundle xgboost DLLs and all package data files
binaries = collect_dynamic_libs('xgboost')

# --- Robust XGBoost VERSION fix ---
# Some conda builds of xgboost omit the VERSION file from the package directory,
# so collect_data_files() silently skips it. XGBoost reads VERSION at import time
# (via open(__file__/../VERSION) or importlib.resources), causing FileNotFoundError
# in the frozen app. Fix: always guarantee the file exists in the bundle.
import xgboost as _xgb
_xgb_ver        = getattr(_xgb, '__version__', '2.0.0')
_xgb_pkg_dir    = os.path.dirname(os.path.abspath(_xgb.__file__))
_xgb_ver_src    = os.path.join(_xgb_pkg_dir, 'VERSION')

if not os.path.exists(_xgb_ver_src):
    # VERSION missing from conda build — create one; filename MUST be 'VERSION'
    # (PyInstaller preserves the source basename in the bundle)
    _tmp_dir = os.path.join(SPECPATH, '_build_tmp')
    os.makedirs(_tmp_dir, exist_ok=True)
    _xgb_ver_src = os.path.join(_tmp_dir, 'VERSION')
    with open(_xgb_ver_src, 'w', encoding='ascii') as _fv:
        _fv.write(_xgb_ver)

# Strip any VERSION already collected (avoid duplicates), then add ours
xgboost_datas = [d for d in collect_data_files('xgboost')
                 if os.path.basename(d[0]) != 'VERSION']
xgboost_datas.append((_xgb_ver_src, 'xgboost'))


a = Analysis(
    [os.path.join(src_dir, 'main.py')],
    pathex=[src_dir],
    binaries=binaries,
    datas=[
        (os.path.join(src_dir, 'models'), 'models'),
        (icon_dir, 'icon'),
    ] + xgboost_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPECPATH, 'rthooks', 'rthook_xgboost_version.py')],
    excludes=['PIL._avif'],
    noarchive=False,
    optimize=0,
)
# Remove PIL._avif binary if it ended up in binaries (decompression workaround)
a.binaries = [b for b in a.binaries if 'PIL/_avif' not in b[0] and 'PIL\\_avif' not in b[0]]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CMSMARK',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX compression can corrupt bundled DLLs (Pillow/matplotlib native libs)
    # on some Windows setups, producing a build that fails to launch.
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[os.path.join(icon_dir, 'app_icon.ico')],
)
