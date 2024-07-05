How to set and use plotting machinery in CMSSW_14_0_0_pre3 (or later) on lxplus (el9 OS)
for making plots and submitting them to /TMP folder of
/eos/project/c/cmsweb/www/hcal-sw-validation/TMP
visible on the web as
https://cms-docs.web.cern.ch/hcal-sw-validation/?dir=TMP

NB: HCAL Validation expert is supposed revise/eamine the plots and then
to move the folders from /TMP to an actual destination deirectory. 

Eample uses CMSSW_14_0_0_pre2 vs CMSSW_14_0_0_pre1 comparison
with the relevant DQM files fetched from the repository directories on
https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/

Target relaease: CMSSW_14_0_0_pre2 , target GT: 133X_mcRun3_2023_realistic_v3
Reference relaease: CMSSW_14_0_0_pre1, reference GT: 133X_mcRun3_2023_realistic_v3
(GTs can be different or the same)

NB: attention should be payed to the versions of the DQM files,
there could be additional strings at the end of their names,
like a version number "vN" (v1 in most of the cases)
or additional special strings, like "STD-v1",


Login to lxplus (=lxplus9)
List which recent CMSSW versions are available:
$scram list --all CMSSW | grep 14_0_0

and create local CMSSW release area:

$export SCRAM_ARCH="el9_amd64_gcc12"  (for tcsh shell: setenv SCRAM_ARCH el9_amd64_gcc12)
$cmsrel CMSSW_14_0_0_pre3 (for instance) or 14_0_0 (when it's available)
$cd CMSSW_14_0_0/src/
$cmsenv
$
$git cms-addpkg Validation/CaloTowers (if all the updated are already available in CMSSW)
$git cms-merge-topic abdoulline:HCAL_RelVal_update (if the branch with updates is not yet merged into CMSSW)
$scram b
$cd Validation/CaloTowers/test/macros/
$make
$voms-proxy-init --voms cms
$export X509_USER_PROXY=/tmp/x509up_uid -u

(1) for making.uploding "2023" (Run3) noPU/PU plots:

Update skProc1_2023.sh  accordinly (using actual target and reference versions): 
./RelValHarvest_2023.py -M CMSSW_14_0_0_pre2
./RelValHarvest_2023.py -M CMSSW_14_0_0_pre1
And execute it (it will download and rename DQM files)
$./skProc1_2023.sh

Update skProc2_2023.sh
to use actual DQM files + change username (replacing "ykazhyka" with your own)
nopu_new="1400pre2_133X_mcRun3_2023_realistic_v3_STD-v1"
nopu_old="1400pre1_133X_mcRun3_2023_realistic_v3-v1"
pu_new="1400pre2_PU_133X_mcRun3_2023_realistic_v3_STD_PU-v3"
pu_old="1400pre1_PU_133X_mcRun3_2023_realistic_v3-v1"

Execute it:
$./skProc2_2023.sh

(2) Similar steps for producing Phase2 (2026D...) noPU/PU plots
(after updating their content)

$./skProc1_Phase2.sh
$./skProc2_Phase2.sh