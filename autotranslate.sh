#!/bin/bash

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate miniconda3-latest/envs/sonitr
python autotranslate.py $1 $2 $3
