from setuptools import setup, find_packages

setup(
    name = "real-deepfakes-detect",
    version = "0.0.1",
    description='Realword DeepFakes Detection',
    long_description='Realword DeepFakes Detection',
    license = "MIT Licence",

    # author='vg450',
    # author_email='gward@python.net',
    url='https://github.com/ThreeCatsLoveFish/RDD',

    packages = find_packages(),
    include_package_data = False,
    install_requires = [
        'pytorchvideo==0.1.5',
        'decord==0.6.0',
        'omegaconf==2.2.2',
        'torch==1.8.1',
        'torchvision==0.9.1',
        'opencv-python==4.5.5.64',
        'scikit-learn==1.0.2',
        'easydict',
        'einops',
        'requests',
        'thop',
        'seaborn',
        'pandas',
    ],

    scripts = [],
    entry_points = {
        'console_scripts': [
            'rdd_inference = inference:main'
        ]
    }
)
