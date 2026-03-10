# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('sklearn')


a = Analysis(
    ['E:\\project\\Dorsa\\cmsmark_program\\src\\main.py'],
    pathex=['E:\\project\\Dorsa\\cmsmark_program\\src'],
    binaries=[],
    datas=[('E:\\project\\Dorsa\\cmsmark_program\\src\\models', 'models'), ('E:\\project\\Dorsa\\cmsmark_program\\icon', 'icon')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['E:\\project\\Dorsa\\cmsmark_program\\icon\\app_icon.ico'],
)
