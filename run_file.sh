#!/bin/bash
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="$CWD/pokemonpython:$PYTHONPATH"
echo $PYTHONPATH
python ${1}