#!/bin/csh
setenv num ${1}

echo '===> num.'  $num 

setenv name pi50

setenv MYWORKDIR $LS_SUBCWD

setenv MYOUT ${MYWORKDIR}
#----------------
cd ${MYWORKDIR}
eval `scramv1 runtime -csh`
cp ${MYWORKDIR}/${name}_${num}.py   ${WORKDIR}/conf.py
#
cd ${WORKDIR}
cp /afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/620/mc_pi50_eta05.root mc.root
echo ${WORKDIR}

cmsRun  conf.py > & ${name}_${num}.log
#---------------------------------------------------------------
 rfcp   ${name}_${num}.log      ${MYWORKDIR}/.
 rfcp   output.root             ${MYOUT}/${name}_${num}.root
