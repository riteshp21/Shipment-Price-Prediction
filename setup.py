from setuptools import find_packages,setup
from typing import List


requirement_file_path  ='requirements.txt'
REMOVE_PACKAGE ='-e .'

def get_requirements()->List[str]:
    with open(requirement_file_path) as requirement_file:
        requirement_list = requirement_file.readline()
        requirement_list = [requirement.replace('\n','') for requirement in requirement_list]
        if REMOVE_PACKAGE in requirement_list:
            requirement_list.remove(REMOVE_PACKAGE)

        return requirement_list


setup(
    name='ShipmentPricePrediction',
    version='1.0',
    description='Shipment Price Prediction',
    author='Ritesh',
    packages=find_packages(),
    install_require =get_requirements()
)


