#!/bin/csh

setenv num ${1}
echo '===> num.'  ${num} 
echo ' ' 

setenv WORKDIR ${PWD}
echo '===> Local working dir ' ${WORKDIR}

setenv name pi50

setenv MYWORKDIR  ${2}  
cd ${MYWORKDIR}

echo ' '
echo '===> Remote submission dir ' ${MYWORKDIR}
echo ' '

eval `scramv1 runtime -csh`

#------------------

cd ${WORKDIR} 

cp ${MYWORKDIR}/${name}_${num}.py  conf.py

xrdcp /afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/620/mc_pi50_eta05.root  mc.root

cmsRun conf.py > & output.log

ls -lrt

xrdcp output.root ${MYWORKDIR}/${name}_${num}.root
xrdcp output.log  ${MYWORKDIR}/${name}_${num}.log
