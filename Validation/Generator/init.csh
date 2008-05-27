
#set PYTHON=/uscmst1/prod/sw/cms/slc4_ia32_gcc345/external/python/2.4.2-CMS10

#unsetenv PYTHONPATH

#source  ${PYTHON}/etc/profile.d/init.csh

setenv VTOOLS_ROOT ${PWD}

setenv PYTHONPATH $VTOOLS_ROOT/python

#setenv ROOTSYS /uscmst1/prod/sw/cms/slc4_ia32_gcc345/lcg/root/5.14.00g-CMS8

setenv PATH ${PATH}:$ROOTSYS/bin

setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:$ROOTSYS/lib


