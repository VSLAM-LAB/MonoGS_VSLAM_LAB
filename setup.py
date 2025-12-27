from setuptools import setup, find_packages

setup(
    name="monogs",
    version="1.0.0",
    description="MonoGS Visual SLAM system",
    packages=find_packages(),
    py_modules=["vslamlab_monogs_mono", "vslamlab_monogs_rgbd" ], 
    install_requires=[
        "numpy",
        "torch",
    ],
    include_package_data=True,
    package_data={
        'gui.gl_render': ['shaders/*.glsl'],
    },
    entry_points={
        "console_scripts": [
            "vslamlab_monogs_mono = vslamlab_monogs_mono:main",
            "vslamlab_monogs_rgbd = vslamlab_monogs_rgbd:main",
        ],
    },
)