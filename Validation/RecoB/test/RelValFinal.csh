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

mkdir TTbar_${val_rel}_vs_${ref_rel}_MC
mkdir TTbar_${val_rel}_vs_${ref_rel}_Startup
mkdir TTbar_${val_rel}_vs_${ref_rel}_FastSim
mkdir QCD_${val_rel}_vs_${ref_rel}_MC
mkdir QCD_${val_rel}_vs_${ref_rel}_Startup

cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REFREL/${ref_rel}/g | sed -e s/VALREL/${val_rel}/g | sed -e s/SAMPLE/TTbar_MC/g > TTbar_${val_rel}_vs_${ref_rel}_MC/ValidationBTag_catprod.xml
cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REFREL/${ref_rel}/g | sed -e s/VALREL/${val_rel}/g | sed -e s/SAMPLE/TTbar_Startup/g > TTbar_${val_rel}_vs_${ref_rel}_Startup/ValidationBTag_catprod.xml
cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REFREL/${ref_rel}/g | sed -e s/VALREL/${val_rel}/g | sed -e s/SAMPLE/TTbar_FastSim/g > TTbar_${val_rel}_vs_${ref_rel}_FastSim/ValidationBTag_catprod.xml
cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REFREL/${ref_rel}/g | sed -e s/VALREL/${val_rel}/g | sed -e s/SAMPLE/QCD_MC/g > QCD_${val_rel}_vs_${ref_rel}_MC/ValidationBTag_catprod.xml
cat ValidationBTag_Template.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REFREL/${ref_rel}/g | sed -e s/VALREL/${val_rel}/g | sed -e s/SAMPLE/QCD_Startup/g > QCD_${val_rel}_vs_${ref_rel}_Startup/ValidationBTag_catprod.xml

cd TTbar_${val_rel}_vs_${ref_rel}_MC
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF
rm cuy.root
rm ValidationBTag_catprod.xml
cp /afs/cern.ch/cms/btag/www/validation/img/0_leg*.gif .
cd ..

cd TTbar_${val_rel}_vs_${ref_rel}_Startup
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

cd QCD_${val_rel}_vs_${ref_rel}_MC
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

mkdir $finaldir
mv *_${val_rel}_vs_${ref_rel}_* $finaldir

echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_TTbar_${val_rel}_vs_${ref_rel}_MC.html'">'TTbar_${val_rel}_vs_${ref_rel}_MC'</a><br>' > index.html
echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_TTbar_${val_rel}_vs_${ref_rel}_Startup.html'">'TTbar_${val_rel}_vs_${ref_rel}_Startup'</a><br>' >> index.html
echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_TTbar_${val_rel}_vs_${ref_rel}_FastSim.html'">'TTbar_${val_rel}_vs_${ref_rel}_FastSim'</a><br>' >> index.html
echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_QCD_${val_rel}_vs_${ref_rel}_MC.html'">'QCD_${val_rel}_vs_${ref_rel}_MC'</a><br>' >> index.html
echo '<a href="https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/index_RecoB_'${finaldir}_QCD_${val_rel}_vs_${ref_rel}_Startup.html'">'QCD_${val_rel}_vs_${ref_rel}_Startup'</a><br>' >> index.html

mv index.html /afs/cern.ch/cms/btag/www/validation/${finaldir}_topdir.html

echo "https://cms-btag-validation.web.cern.ch/cms-btag-validation/validation/${finaldir}_topdir.html" >& webpage.txt 
