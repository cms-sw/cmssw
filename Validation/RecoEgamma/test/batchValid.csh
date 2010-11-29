#!/bin/csh
setenv sim    ${1}
setenv type   ${2}
setenv sample ${3} 


echo '===> simulation' $sim
echo '===> type'       $type
echo '===> sample.'  $sample 


if ( $sample == SingleGammaPt10 ) then
setenv outFileName SingleGammaPt10
else if (  $sample == SingleGammaPt35 ) then
setenv outFileName SingleGammaPt35
else if (  $sample == SingleGammaFlatPt10To100 ) then
setenv outFileName SingleGammaFlatPt10To100
else if (  $sample ==  H130GGgluonfusion ) then
setenv outFileName H130GGgluonfusion
else if (  $sample == PhotonJets_Pt_10 ) then
setenv outFileName  PhotonJets_Pt_10
else if (  $sample == QCD_Pt_20_30 ) then
setenv outFileName  QCD_Pt_20_30
else if (  $sample == QCD_Pt_80_120 ) then
setenv outFileName  QCD_Pt_80_120
endif

if ($sim == Full ) then
setenv confName  ${type}Validator
else if ( $sim == Fast ) then
setenv confName  PhotonValidatorFastSim
endif

setenv MYWORKDIR /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_3_9_4/src/Validation/RecoEgamma/test

echo ${MYWORKDIR}
setenv MYOUT ${MYWORKDIR}
#----------------
cd ${MYWORKDIR}
eval `scramv1 runtime -csh`

if ( $sim == Full ) then
cp ${MYWORKDIR}/${confName}_${sample}.py    ${WORKDIR}/conf.py
else if ( $sim == Fast ) then
cp ${MYWORKDIR}/${confName}.py    ${WORKDIR}/conf.py
endif


cd ${WORKDIR}
echo ${WORKDIR}

cmsRun  conf.py > & ${outFileName}.log
#---------------------------------------------------------------


if ( $sim == Full ) then
 rfcp   ${outFileName}.log             ${MYOUT}/${outFileName}.log

 rfcp   PhotonValidationRelVal394_${outFileName}.root            ${MYOUT}/.
 rfcp   ConversionValidationRelVal394_${outFileName}.root        ${MYOUT}/.
else if ( $sim == Fast ) then
 rfcp   ${outFileName}.log             ${MYOUT}/${outFileName}_FastSim.log
rfcp   PhotonValidationRelVal394_${outFileName}_FastSim.root            ${MYOUT}/.
endif
