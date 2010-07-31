#! /bin/csh

setenv URL $1

if ($#argv > 0) then
echo Downloading file $URL

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.csh
voms-proxy-init
curl -O -L --capath $X509_CERT_DIR --key $X509_USER_PROXY --cert $X509_USER_PROXY $URL

else
echo Syntax:
echo ./getHarvestedHistos.csh https://etc.
echo where the URL must be taken from https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/RelVal/
endif
