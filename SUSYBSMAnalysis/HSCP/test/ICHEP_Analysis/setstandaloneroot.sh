#!/bin/sh
#
arch=slc5_ia32_gcc434
export CMS_PATH=/uscmst1/prod/sw/cms

export PYTHONDIR=/uscmst1/prod/sw/cms/slc5_ia32_gcc434/external/python/2.6.4-cms6
export PATH=${PYTHONDIR}/bin:/uscmst1b_scratch/lpc1/lpcphys/jchen/root/root_v5.30.02/bin:$PATH
export ROOTSYS=/uscmst1b_scratch/lpc1/lpcphys/jchen/root/root_v5.30.02
export PYTHONPATH=${ROOTSYS}/lib:${PYTHONPATH}
export LD_LIBRARY_PATH=${PYTHONDIR}/lib:${CMS_PATH}/$arch/external/gcc/4.3.4/lib:${ROOTSYS}:${ROOTSYS}/lib:${LD_LIBRARY_PATH}
export ROOT_INCLUDE=${ROOTSYS}/include
