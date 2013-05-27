#!/bin/tcsh

#Check for correct number of arguments
if ($#argv<2) then
    echo "Script needs 2 arguments for comparison"
    exit
endif

#Check to see if the CMS environment is set up
if ($?CMSSW_BASE != 1) then
    echo "CMS environment not set up"
    exit
endif

set val_rel=$1
set ref_rel=$2
set cur_dir=`pwd`
set finaldir=$CMSSW_VERSION

mkdir TTbar_${val_rel}_vs_${ref_rel}_Startup
mkdir TTbar_${val_rel}_vs_${ref_rel}_Startup_PU
mkdir TTbar_${val_rel}_vs_${ref_rel}_FastSim
mkdir QCD_${val_rel}_vs_${ref_rel}_Startup
mkdir FastSim_TTbar_${val_rel}_vs_TTbar_${val_rel}
mkdir PU_TTbar_${val_rel}_vs_TTbar_${val_rel}

cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REFREL/${ref_rel}/g | sed -e s/VALREL/${val_rel}/g | sed -e s/SAMPLE/TTbar_Startup/g > TTbar_${val_rel}_vs_${ref_rel}_Startup/ValidationBTag_catprod.xml
cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REFREL/${ref_rel}/g | sed -e s/VALREL/${val_rel}/g | sed -e s/SAMPLE/TTbar_Startup_PU/g > TTbar_${val_rel}_vs_${ref_rel}_Startup_PU/ValidationBTag_catprod.xml
cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REFREL/${ref_rel}/g | sed -e s/VALREL/${val_rel}/g | sed -e s/SAMPLE/TTbar_FastSim/g > TTbar_${val_rel}_vs_${ref_rel}_FastSim/ValidationBTag_catprod.xml
cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REFREL/${ref_rel}/g | sed -e s/VALREL/${val_rel}/g | sed -e s/SAMPLE/QCD_Startup/g > QCD_${val_rel}_vs_${ref_rel}_Startup/ValidationBTag_catprod.xml
cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/SAMPLE_REFREL/TTbar_Startup_${val_rel}/g | sed -e s/SAMPLE_VALREL/TTbar_FastSim_${val_rel}/g  | sed -e s/VALREL_SAMPLE/${val_rel}_TTbar_FastSim/g | sed -e s/REFREL_SAMPLE/${val_rel}_TTbar_Startup/g | sed -e s/SAMPLE/TTbar/g | sed -e s/VALREL/${val_rel}/g > FastSim_TTbar_${val_rel}_vs_TTbar_${val_rel}/ValidationBTag_catprod.xml
cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/SAMPLE_REFREL/TTbar_Startup_${val_rel}/g | sed -e s/SAMPLE_VALREL/TTbar_Startup_PU${val_rel}/g  | sed -e s/VALREL_SAMPLE/${val_rel}_TTbar_Startup_PU/g | sed -e s/REFREL_SAMPLE/${val_rel}_TTbar_Startup/g | sed -e s/SAMPLE/TTbar/g | sed -e s/VALREL/${val_rel}/g > PU_TTbar_${val_rel}_vs_TTbar_${val_rel}/ValidationBTag_catprod.xml

cd TTbar_${val_rel}_vs_${ref_rel}_Startup
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
cd ..

cd TTbar_${val_rel}_vs_${ref_rel}_Startup_PU
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
cd ..

cd TTbar_${val_rel}_vs_${ref_rel}_FastSim
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
cd ..

cd QCD_${val_rel}_vs_${ref_rel}_Startup
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
cd ..

cd FastSim_TTbar_${val_rel}_vs_TTbar_${val_rel}
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
#rm ValidationBTag_catprod.xml
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
cd ..

cd PU_TTbar_${val_rel}_vs_TTbar_${val_rel}
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
cd ..

mkdir $finaldir
mv *_${val_rel}_vs_${ref_rel}_*                 $finaldir/
mv FastSim_TTbar_${val_rel}_vs_TTbar_${val_rel} $finaldir/
mv PU_TTbar_${val_rel}_vs_TTbar_${val_rel}      $finaldir/

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_TTbar_${val_rel}_vs_${ref_rel}_Startup.html'">'TTbar_${val_rel}_vs_${ref_rel}_Startup'</a><br>' > index.html
echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_TTbar_${val_rel}_vs_${ref_rel}_Startup_PU.html'">'TTbar_${val_rel}_vs_${ref_rel}_Startup_PU'</a><br>' >> index.html
echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_TTbar_${val_rel}_vs_${ref_rel}_FastSim.html'">'TTbar_${val_rel}_vs_${ref_rel}_FastSim'</a><br>' >> index.html
echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_QCD_${val_rel}_vs_${ref_rel}_Startup.html'">'QCD_${val_rel}_vs_${ref_rel}_Startup'</a><br>' >> index.html
echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_FastSim_TTbar_${val_rel}_vs_TTbar_${val_rel}.html'">'FastSim_TTbar_${val_rel}_vs_TTbar_${val_rel}'</a><br>' >> index.html
echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_PU_TTbar_${val_rel}_vs_TTbar_${val_rel}.html'">'PU_TTbar_${val_rel}_vs_TTbar_${val_rel}'</a><br>' >> index.html


mv index.html /afs/cern.ch/cms/btag/www/validation/${finaldir}_topdir.html

echo "https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/${finaldir}_topdir.html" >& webpage.txt 
