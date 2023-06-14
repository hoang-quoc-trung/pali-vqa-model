# Copyright 2023 The PaLi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install PaLi."""

import os
import sys
import setuptools
import logging
import subprocess

logging.getLogger().setLevel(logging.INFO)
logging.info('Installing jax')
subprocess.run(['conda', 'install', 'jax', 'cuda-nvcc', '-c', 'conda-forge', '-c', 'nvidia'])
subprocess.run(['pip', 'install', 'nest_asyncio'])
logging.info('Installing t5x...')
subprocess.run(['python', '-m','pip', 'install', 'src/models/t5x'])
logging.info('Installing vit...')
subprocess.run(['python', '-m','pip', 'install', 'src/models/vit'])

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'pali')
sys.path.append(version_path)
__version__ = '0.0.0'

setuptools.setup(
    name='pali',
    version=__version__,
    description='PaLi: A Python Library for Pretraining and Fine-tuning Transformers',
    url='https://gitlab.com/gradlabs/research/imgcaption',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'jax>=0.4.1',
    ]
)
