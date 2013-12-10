#!/bin/tcsh

#change release version : $1=release to validate, $2=reference release, $3=working directory (where is lacated yours CMSSW instances), $4=release name for DQM inputs to validate, $5=the same for reference DQM inputs
valrel=$1
refrel=$2
workdir=$3
valdir=$4
refdir=$5

mkdir TTbar_${valrel}_vs_${refrel}_Startup
mkdir TTbar_${valrel}_vs_${refrel}_Startup_PU
mkdir TTbar_${valrel}_vs_${refrel}_FastSim
mkdir QCD_${valrel}_vs_${refrel}_Startup
mkdir FastSim_TTbar_${valrel}_vs_TTbar_${valrel}

cd TTbar_${valrel}_vs_${refrel}_Startup
plotFactory.py -b -f ${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test/BTagRelVal_TTbar_Startup_${valrel}.root -F ${workdir}/CMSSW_${refdir}/src/Validation/RecoB/test/BTagRelVal_TTbar_Startup_${refrel}.root -r ${valrel} -R ${refrel} -s TTbar_Startup -S TTbar_Startup
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}_*_allTaggers*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${refrel}_*_allTaggers*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..

cd TTbar_${valrel}_vs_${refrel}_Startup_PU
plotFactory.py -b -f ${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test/BTagRelVal_TTbar_Startup_PU_${valrel}.root -F ${workdir}/CMSSW_${refdir}/src/Validation/RecoB/test/BTagRelVal_TTbar_Startup_PU_${refrel}.root -r ${valrel} -R ${refrel} -s TTbar_Startup_PU -S TTbar_Startup_PU
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}_*_allTaggers*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${refrel}_*_allTaggers*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..

cd TTbar_${valrel}_vs_${refrel}_FastSim
plotFactory.py -b -f ${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test/BTagRelVal_TTbar_FastSim_${valrel}.root -F ${workdir}/CMSSW_${refdir}/src/Validation/RecoB/test/BTagRelVal_TTbar_FastSim_${refrel}.root -r ${valrel} -R ${refrel} -s TTbar_FastSim -S TTbar_FastSim
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}_*_allTaggers*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${refrel}_*_allTaggers*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..
 

cd QCD_${valrel}_vs_${refrel}_Startup
plotFactory.py -b -f ${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test/BTagRelVal_QCD_Startup_${valrel}.root -F ${workdir}/CMSSW_${refdir}/src/Validation/RecoB/test/BTagRelVal_QCD_Startup_${refrel}.root -r ${valrel} -R ${refrel} -s QCD_Startup -S QCD_Startup
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}_*_allTaggers*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${refrel}_*_allTaggers*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..

cd FastSim_TTbar_${valrel}_vs_TTbar_${valrel}
plotFactory.py -b -f ${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test/BTagRelVal_TTbar_FastSim_${valrel}.root -F ${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test/BTagRelVal_TTbar_Startup_${valrel}.root -r ${valrel} -R ${valrel} -s TTbar_FastSim -S TTbar_Startup
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}*FastSim*_allTaggers*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${valrel}*Startup*_allTaggers*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..

mkdir CMSSW_${valdir}
mv *_${valrel}_vs_${refrel}_*                 CMSSW_${valdir}/
mv FastSim_TTbar_${valrel}_vs_TTbar_${valrel} CMSSW_${valdir}/

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_TTbar_${valrel}_vs_${refrel}_Startup.html'">'TTbar_${valrel}_vs_${refrel}_Startup'</a><br>' >> index.html

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_TTbar_${valrel}_vs_${refrel}_Startup_PU.html'">'TTbar_${valrel}_vs_${refrel}_Startup_PU'</a><br>' >> index.html

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_TTbar_${valrel}_vs_${refrel}_FastSim.html'">'TTbar_${valrel}_vs_${refrel}_FastSim'</a><br>' >> index.html

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_QCD_${valrel}_vs_${refrel}_Startup.html'">'QCD_${valrel}_vs_${refrel}_Startup'</a><br>' >> index.html

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_FastSim_TTbar_${valrel}_vs_TTbar_${valrel}.html'">'FastSim_TTbar_${valrel}_vs_TTbar_${valrel}'</a><br>' >> index.html



#mv index.html /afs/cern.ch/cms/btag/www/validation/CMSSW_${valdir}_topdir.html


echo "https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/CMSSW_${valdir}_topdir.html" >& webpage.txt 
