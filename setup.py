from setuptools import find_packages, setup
from typing import List

hyphen='-e .'

def get_requirement(file_path:str)->List[str]:
    '''This function will return the requirments.txt as a list
    '''
    requirements=[]
    with open(file_path) as req_file:
        req=req_file.readlines()
        requirements=[x.replace('\n','') for x in req]
    if hyphen in requirements:
        requirements.remove(hyphen)
    return requirements
 
setup(
    name='end_to_end_project',
    version='0.0.1',
    author='Sathwik',
    author_email='sathwikmethari@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)