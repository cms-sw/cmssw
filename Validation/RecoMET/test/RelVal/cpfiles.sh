#! /bin/bash

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh;
voms-proxy-init;

swversion=4_2_0

cond=MC_42_V9_FastSim-v1; 

URLHEAD=https://cmsweb.cern.ch/dqm/dev/data/browse/Development/RelVal/CMSSW_4_2_x;
URLHEAD=https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/RelVal/CMSSW_4_2_x;

cd /tmp/wangdy;
mkdir $swversion;
cd $swversion;

for SAMP in  TTbar  QCD_FlatPt_15_3000 ; do
thefile="$URLHEAD"/DQM_V0001_R000000001__RelVal"$SAMP"__CMSSW_"$swversion"-"$cond"__GEN-SIM-DIGI-RECO.root;
echo $thefile;
#wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY $thefile;
wget --no-check-certificate  --private-key ~/.globus/userkey.pem  --certificate ~/.globus/usercert.pem $thefile;
done;


cond=MC_42_V9-v1; ##4_2_0

for SAMP in  QCD_Pt_80_120  TTbar LM1_sfts QCD_FlatPt_15_3000 ; do
thefile="$URLHEAD"/DQM_V0003_R000000001__RelVal"$SAMP"__CMSSW_"$swversion"-"$cond"__DQM.root;
#thefile="$URLHEAD"/DQM_V0001_R000000001__RelVal"$SAMP"__CMSSW_"$swversion"-"$cond"__GEN-SIM-RECO.root;
echo $thefile;
# ## method1: sometimes does not work 
# wget --ca-directory $X509_CERT_DIR/ --certificate=$X509_USER_PROXY --private-key=$X509_USER_PROXY $thefile;

## method2 need to input passwd for each file !
wget --no-check-certificate  --private-key ~/.globus/userkey.pem  --certificate ~/.globus/usercert.pem $thefile;
##mehtod3
#curl -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY $thefile;

done;