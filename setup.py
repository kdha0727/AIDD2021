#nsml: pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

from distutils.core import setup

setup(
    name='nia dm hackathon example',
    version='1.0',
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'scikit-learn',
        'imblearn',
        'pytorch-tabnet',
        'xgboost',
        'smote_variants',
        'lightgbm',
    ]
)

# 1.5-cuda10.1-cudnn7-runtime
# 1.6.0-cuda10.1-cudnn7-runtime
# 1.8.0-cuda11.1-cudnn8-runtime
# 1.9.1-cuda11.1-cudnn8-runtime
