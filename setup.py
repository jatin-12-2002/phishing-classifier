from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """This function returns the list of requirements from requirements.txt file
    """
    requirement_list:List[str] = []
    with open('requirements.txt', 'r') as file:
        for line in file.readlines():
            requirement_list = [line.strip() for line in file.readlines()]
            #requirement_list = file.read().splitlines()
    
    return requirement_list
# Specify the editable installation separately (if needed)
editable_installation = ["-e ."]

setup(
    name='phishing',
    version='0.0.1',
    author='Jatin Sareen',
    author_email="sareenjatin002@gmail.com",
    packages=find_packages(),
    #install_requires=get_requirements()  #["pymongo==4.2.0"]
    install_requires=get_requirements(),  # Production dependencies
    dependency_links=editable_installation,  # Editable installation
)