"""
The setup.py file is an essential part of packaging and distributing python projects.
It is used by setuptools (or disutils in older Python version) to define the confisguration
of your project, such as its metadata, dependencies and more.
"""

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    This function will return list of requirements.
    """

    requirement_list: List[str] = []
    try:
        with open ('requirements.txt', 'r') as file:
            # Read lines from the file
            lines = file.readlines()

            # Process each line
            for line in lines:
                requirement = line.strip()
                # Ignore empty line and -e .
                if requirement and requirement!='-e .':
                    requirement_list.append(requirement)
    
    except FileNotFoundError:
        print('requirements.txt file not found!')

    return requirement_list

setup(
    name='RestaurantRating',
    version='1.0.0',
    description='End-to-End ML pipeline to calculate Restaurant Rating',
    author='Rahul',
    author_email='pymail7789@gmail.com',
    python_requires='>=3.13',
    packages=find_packages(),
    install_requires=get_requirements(),
    extras_require={
        'tracking': [
            "mlflow"
        ]
    },
    entry_points={
        'console_scripts': [
            "train=scripts.cli_run:main"
        ]
    },
)