#! /bin/bash

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh;
voms-proxy-init;
swversion=3_8_3
cond=MC_3XY_V24_FastSim-v1;
URLHEAD=https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/RelVal/CMSSW_3_8_x;

cd /tmp/wangdy;
mkdir $swversion;
cd $swversion;

# for SAMP in  TTbar  QCD_FlatPt_15_3000 ; do
# thefile="$URLHEAD"/DQM_V0001_R000000001__RelVal"$SAMP"__CMSSW_"$swversion"-"$cond"__GEN-SIM-DIGI-RECO.root;
# echo $thefile;
# wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY $thefile;
# done;

cond=MC_38Y_V9-v1;
for SAMP in  QCD_Pt_80_120 QCD_Pt_3000_3500 Wjet_Pt_80_120 TTbar LM1_sfts QCD_FlatPt_15_3000 ; do
thefile="$URLHEAD"/DQM_V0001_R000000001__RelVal"$SAMP"__CMSSW_"$swversion"-"$cond"__GEN-SIM-RECO.root;
echo $thefile;
## method1: sometimes does not work 
wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY $thefile;

##  method2 need to input passwd for each file !
#wget --no-check-certificate  --private-key ~/.globus/userkey.pem  --certificate ~/.globus/usercert.pem $thefile;


done;