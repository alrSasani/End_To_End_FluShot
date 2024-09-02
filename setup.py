from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e . '

def get_requirments(req_file):
    with open(req_file,'r') as file_obj:
        requirments = file_obj.readlines()
    requirments = [req.replace('\n',' ') for req in requirments]
    if HYPHEN_E_DOT in requirments:
        requirments.remove(HYPHEN_E_DOT)
    print(requirments,end=' ')
    return requirments


setup(
    name="flu-shot-prediction",
    version = "0.0.1",
    # description="***",
    author_email='alr.sasani@gmail.com',
    author='Alireza',
    packages=find_packages(),
    install_requires=get_requirments('requirments.txt')
)