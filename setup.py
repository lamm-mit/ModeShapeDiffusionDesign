from setuptools import setup, find_packages

# Function to read requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as req_file:
        return req_file.read().splitlines()

setup(
    name='VibeGen',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    description='VibeGen: End-to-end de novo protein generation targeting normal mode vibrations using a language diffusion model duo',
    author='Bo Ni',
    url='https://github.com/lamm-mit/ModeShapeDiffusionDesign',
)
