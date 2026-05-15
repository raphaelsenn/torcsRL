from setuptools import setup, find_packages

setup(
    name="torcsrl",
    version="0.0.1",
    description="OpenAI Gym wrapper for TORCS racing simulator",
    author="gym_torcs contributors",
    packages=find_packages(),
    install_requires=[
        "gymnasium==1.3.0",
        "numpy>=2.2,<2.5",
    ],
    python_requires=">=3.11",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)