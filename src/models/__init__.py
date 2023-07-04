import os
import sys

t5x_modules = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t5x")
sys.path.append(t5x_modules)

vit_modules = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vit_jax")
sys.path.append(vit_modules)