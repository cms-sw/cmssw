#!/bin/csh
setenv num ${1}

echo '===> num.'  $num 

if ( $num == 1 ) then
setenv outFileName SingleGammaPt10
else if (  $num == 2 ) then
setenv outFileName SingleGammaPt35
else if (  $num == 3 ) then
setenv outFileName H130GGgluonfusion
else if (  $num == 4 ) then
setenv outFileName  QCD_Pt_80_120
endif

setenv name  PhotonValidator_cfg

setenv MYWORKDIR /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/slc5_ia32_gcc434/CMSSW_3_4_0_pre5/src/Validation/RecoEgamma/test

echo ${MYWORKDIR}

setenv MYOUT ${MYWORKDIR}
#----------------
cd ${MYWORKDIR}
eval `scramv1 runtime -csh`
cp ${MYWORKDIR}/${name}_${num}.py    ${WORKDIR}/conf.py


#
cd ${WORKDIR}
echo ${WORKDIR}

cmsRun  conf.py > & ${outFileName}.log
#---------------------------------------------------------------
 rfcp   ${outFileName}.log             ${MYOUT}/.
 rfcp   PhotonValidationRelVal340pre5_${outFileName}.root            ${MYOUT}/.
