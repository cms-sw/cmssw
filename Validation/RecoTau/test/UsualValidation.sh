#!/bin/bash

if [ -f EventSource_QCD_RECO_cff.py ]; then

else
    cp EventSource_ZTT_RECO_cff.py EventSource_QCD_RECO_cff.py
fi

`./query_dbs.sh`
`./RunValidation_cfg.py`
`./RunValidation_cfg.py eventType=QCD`

source UtilityCommands.sh
cd TauID/ZTT_recoFiles/
`Summarize`
label=`echo $CMSSW_VERSION | awk -FCMSSW_ '{print $2}'`
`Compare compareTo=$PastResults/CMSSW_3_9_4/TauID/ZTT_recoFiles/ testLabel=$label referenceLabel=3_9_4`
`Compare compareTo=../QCD_recoFiles/ testLabel=ZTT referenceLabel=QCD scale=smartlog`
`BuildWebpage`
if [ -z "$DISPLAY" ]; then
    echo "Display not set. Impossible to open the page"
else
    `firefox index.html`
fi

echo 'Do you want to submit the result? [y/n]'
read answer
if [ $answer = 'y' ]; then
    `SubmitResults`
else
    exit
fi
