#!/bin/bash

wregexp=$1
if [ -z $wregexp ]; then
    echo Search pattern for datasets not specified. Defaulting...
    wregexp="/RelValSingleMuPt*/GEN*RECO"
fi
echo WREGEXP=$wregexp

cmssw_release=$2
if [ -z $cmssw_release ]; then
    echo CMSSW release not specified. Check \$CMSSW_VERSION...
    if [ -n $CMSSW_VERSION ]; then
        cmssw_release=$CMSSW_VERSION
        echo CMSSW_VERSION=$cmssw_release. Using this...
    else
        echo CMSSW_VERSION not set. Exiting...
    fi
else
    echo Using CMSSW release $cmssw_release...
fi

cfgpysIndex=0

cmd="dbsql \"find dataset where dataset like $wregexp and release = $cmssw_release\""
#echo $cmd
for line in `eval $cmd`; do
    if [[ $line =~ '^/' ]]; then
        dataset=$line
        echo Fetching dataset $dataset...
        field1=`echo $dataset | awk -F'/' '{print $2}'`
        field2=`echo $dataset | awk -F'/' '{print $3}'`
        field3=`echo $dataset | awk -F'/' '{print $4}'`
        workflow=$dataset

        # create cfg.py for this dataset
        cfgpy="MuonIdValToME__${field1}__${field2}__${field3}.py"
        cfgpys[$cfgpysIndex]=$cfgpy
        cat <<EOF >$cfgpy
import FWCore.ParameterSet.Config as cms

process = cms.Process("MUONIDVALtoME")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("DQMServices.Components.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.dqmSaver.convention = "Offline" # "RelVal"
process.dqmSaver.workflow = "$workflow"
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

process.p = cms.Path(process.EDMtoMEConverter*process.dqmSaver)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
EOF

        cmd="dbsql \"find file where dataset = $dataset\""
        #echo $cmd

        for file in `eval $cmd`; do
            if [[ $file =~ '^/' ]]; then
                echo "\"$file\"," >>$cfgpy
            fi
        done
echo '    )' >>$cfgpy
echo ')' >>$cfgpy
    fi

    cfgpysIndex=$cfgpysIndex+1
done

echo "Running ${#cfgpys[@]} cmsRun jobs..."
for cfgpy in ${cfgpys[*]}
do
    echo "cmsRun $cfgpy"
    cmsRun $cfgpy
done
