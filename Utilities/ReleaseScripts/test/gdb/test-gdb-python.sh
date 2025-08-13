#!/bin/bash -ex
gdb_python=$(gdb -q -ex 'python exec("import sys;print(sys.executable)")' -ex quit)
cmssw_python=$(scram tool tag python3 PYTHON3_BASE)/bin
cmssw_python=$(realpath ${cmssw_python})/python
if [ "${gdb_python}" != "${cmssw_python}" ] ; then
  echo "python used by GDB and CMSSW are not same"
  exit 1
else
  echo "CMSSW/GDB python OK"
fi
