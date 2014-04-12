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


cat ValidationBTag_Template.xml | sed -e s%REFDIR%${workdir}/CMSSW_${refdir}/src/Validation/RecoB/test%g | sed -e s%VALDIR%${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test%g | sed -e s/REFREL/${refrel}/g | sed -e s/VALREL/${valrel}/g | sed -e s/SAMPLE/TTbar_Startup/g > TTbar_${valrel}_vs_${refrel}_Startup/ValidationBTag_catprod.xml

cat ValidationBTag_Template.xml | sed -e s%REFDIR%${workdir}/CMSSW_${refdir}/src/Validation/RecoB/test%g | sed -e s%VALDIR%${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test%g | sed -e s/REFREL/${refrel}/g | sed -e s/VALREL/${valrel}/g | sed -e s/SAMPLE/TTbar_Startup_PU/g > TTbar_${valrel}_vs_${refrel}_Startup_PU/ValidationBTag_catprod.xml

cat ValidationBTag_Template.xml | sed -e s%REFDIR%${workdir}/CMSSW_${refdir}/src/Validation/RecoB/test%g | sed -e s%VALDIR%${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test%g | sed -e s/REFREL/${refrel}/g | sed -e s/VALREL/${valrel}/g | sed -e s/SAMPLE/TTbar_FastSim/g > TTbar_${valrel}_vs_${refrel}_FastSim/ValidationBTag_catprod.xml

cat ValidationBTag_Template.xml | sed -e s%REFDIR%${workdir}/CMSSW_${refdir}/src/Validation/RecoB/test%g | sed -e s%VALDIR%${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test%g | sed -e s/REFREL/${refrel}/g | sed -e s/VALREL/${valrel}/g | sed -e s/SAMPLE/QCD_Startup/g > QCD_${valrel}_vs_${refrel}_Startup/ValidationBTag_catprod.xml

cat ValidationBTag_Template.xml | sed -e s%REFDIR%${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test%g | sed -e s%VALDIR%${workdir}/CMSSW_${valdir}/src/Validation/RecoB/test%g | sed -e s/SAMPLE_REFREL/TTbar_Startup_${valrel}/g | sed -e s/SAMPLE_VALREL/TTbar_FastSim_${valrel}/g  | sed -e s/VALREL_SAMPLE/${valrel}_TTbar_Startup/g | sed -e s/REFREL_SAMPLE/${valrel}_TTbar_FastSim/g | sed -e s/SAMPLE/TTbar/g | sed -e s/VALREL/${valrel}/g > FastSim_TTbar_${valrel}_vs_TTbar_${valrel}/ValidationBTag_catprod.xml

cd TTbar_${valrel}_vs_${refrel}_Startup
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
rm *not*
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}_*_ALL_tagger*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}_*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${refrel}_*_ALL_tagger*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}_*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..

cd TTbar_${valrel}_vs_${refrel}_Startup_PU
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
rm *not*
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}_*_ALL_tagger*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}_*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${refrel}_*_ALL_tagger*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}_*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..

cd TTbar_${valrel}_vs_${refrel}_FastSim
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
rm *not*
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}_*_ALL_tagger*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}_*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${refrel}_*_ALL_tagger*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}_*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..
 

cd QCD_${valrel}_vs_${refrel}_Startup
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
rm *not*
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}_*_ALL_tagger*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}_*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${refrel}_*_ALL_tagger*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}_*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..

cd FastSim_TTbar_${valrel}_vs_TTbar_${valrel}
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
#rm ValidationBTag_catprod.xml
rm *not*
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
mv 0_leg1.gif 0_leg4.gif
for File in `ls ${valrel}*FastSim*_ALL_tagger*.gif`; do mv ${File} 1_${File}; done
for File in `ls ${valrel}_*jetPt*.gif`; do mv ${File} 2_${File}; done
for File in `ls ${valrel}*Startup*_ALL_tagger*.gif`; do mv ${File} 3_${File}; done
for File in `ls ${valrel}_*jetEta*.gif`; do mv ${File} 4_${File}; done
cd ..

mkdir CMSSW_${valdir}
mv *_${valrel}_vs_${refrel}_*                 CMSSW_${valdir}/
mv FastSim_TTbar_${valrel}_vs_TTbar_${valrel} CMSSW_${valdir}/

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_TTbar_${valrel}_vs_${refrel}_Startup.html'">'TTbar_${valrel}_vs_${refrel}_Startup'</a><br>' >> index.html

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_TTbar_${valrel}_vs_${refrel}_Startup_PU.html'">'TTbar_${valrel}_vs_${refrel}_Startup_PU'</a><br>' >> index.html

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_TTbar_${valrel}_vs_${refrel}_FastSim.html'">'TTbar_${valrel}_vs_${refrel}_FastSim'</a><br>' >> index.html

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_QCD_${valrel}_vs_${refrel}_Startup.html'">'QCD_${valrel}_vs_${refrel}_Startup'</a><br>' >> index.html

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'CMSSW_${valdir}_FastSim_TTbar_${valrel}_vs_TTbar_${valrel}.html'">'FastSim_TTbar_${valrel}_vs_TTbar_${valrel}'</a><br>' >> index.html



mv index.html /afs/cern.ch/cms/btag/www/validation/CMSSW_${valdir}_topdir.html


echo "https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/CMSSW_${valdir}_topdir.html" >& webpage.txt 
