# -*- mode: python ; coding: utf-8 -*-

from kivy_deps import sdl2, glew

a = Analysis(
    ['main_UI.py', 'Config.py', 'DatabasePage.py', 'Dialog.py', 'IndexPage.py', 'our_atabase.py', 'RefilterPage.py', 'SelectPage.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\algorithms\__init__.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\algorithms\aca.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\algorithms\afsa.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\algorithms\data_structure.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\algorithms\ga.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\algorithms\others_demo.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\algorithms\pso.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\algorithms\sa.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\algorithms\utils.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\database\__init__.py',
    r'D:\文档\人工智能\example\optimal_sample_selection\database\database.py'],
    pathex=[],
    binaries=[],
    datas=[
        (r'D:\文档\人工智能\example\optimal_sample_selection\guipge', 'guipge'),
        (r'D:\文档\人工智能\example\optimal_sample_selection\static', 'static'),
        (r'D:\文档\人工智能\example\optimal_sample_selection\database.json', '.')
    ],
    hiddenimports=[
        'matplotlib', 'numpy', 'pandas', 'sko', 'sko.GA', 'sko.PSO', 'sko.SA',
        'collections', 'itertools', 'os', 'sys', 'json', 'typing', 'math', 'time', 'string', 'random'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0
)

pyz = PYZ(a.pure)

exe = EXE(pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='OSS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='D:\\文档\\人工智能\\example\\optimal_sample_selection\\static\\logo.ico'
)

coll = COLLECT(exe, Tree(r'D:\文档\人工智能\example\optimal_sample_selection\algorithms'),
                Tree(r'D:\文档\人工智能\example\optimal_sample_selection\database'),
               a.binaries,
               a.zipfiles,
               a.datas,
               *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               name='OSS')