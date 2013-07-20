#!/bin/tcsh
#
set arch = slc5_ia32_gcc434
setenv CMS_PATH /uscmst1/prod/sw/cms

setenv PYTHONDIR  /uscmst1/prod/sw/cms/slc5_ia32_gcc434/external/python/2.6.4-cms6
setenv PATH  ${PYTHONDIR}/bin:/uscmst1b_scratch/lpc1/lpcphys/jchen/root/root_v5.30.02/bin:$PATH
setenv ROOTSYS  /uscmst1b_scratch/lpc1/lpcphys/jchen/root/root_v5.30.02
setenv PYTHONPATH  ${ROOTSYS}/lib:${PYTHONPATH}
setenv LD_LIBRARY_PATH  ${PYTHONDIR}/lib:${CMS_PATH}/$arch/external/gcc/4.3.4/lib:${ROOTSYS}:${ROOTSYS}/lib:${LD_LIBRARY_PATH}
setenv ROOT_INCLUDE  ${ROOTSYS}/include
