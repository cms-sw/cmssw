#!/bin/bash
#Script to run RECO and DQM sequences on existing files using cmsDriver.py
#More background information: 
#https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCmsDriver
#https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookDataFormats

hostname
env
voms-proxy-info

#abort on error
set -e
set -x

#number of events to process per job
#passed through condor, but here is a default value
if [ -z "$PERJOB" ]; then
    PERJOB=200
fi

#
#set default conditions - run3 2021
CONDITIONS=auto:phase1_2022_realistic ERA=Run3 GEOM=DB.Extended CUSTOM=
#
#conditions - 2018
#CONDITIONS=auto:phase1_2018_realistic ERA=Run2_2018 GEOM=DB.Extended CUSTOM=
#
#conditions - phase2
#CONDITIONS=auto:phase2_realistic_T15 ERA=Phase2C9 GEOM=Extended2026D49 CUSTOM="--customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000"

#Running with 2 threads allows to use more memory on grid
NTHREADS=8

#Argument parsing
if [ "$#" -ne 3 ]; then
    echo "Must pass exactly 3 arguments: run_relval.sh [QCD|QCDPU|ZEEPU|ZMMPU|TenTauPU|NuGunPU] [reco|dqm] [njob]"
    exit 0
fi

#index of the job is used to keep track of which events / files to process in the reco step
NJOB=$(($3 + 1))

#set CMSSW environment and go to condor work dir
LAUNCHDIR=`pwd`
source /cvmfs/cms.cern.ch/cmsset_default.sh

#this environment variable comes from the condor submit script
cd $CMSSW_BASE
eval `scram runtime -sh`

#define HOME if not defined.
if [ -z "$HOME" ]; then
    export HOME=/tmp
fi

#if the _CONDOR_SCRATCH_DIR is not defined, we are not inside a condor batch job
if [ -z "$_CONDOR_SCRATCH_DIR" ]; then
    cd $LAUNCHDIR
else
    cd $_CONDOR_SCRATCH_DIR
fi

##RelVal samples
if [ "$1" == "QCD" ]; then
    INPUT_FILELIST=${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/tmp/das_cache/QCD_noPU.txt
    NAME=QCD
elif [ "$1" == "QCDPU" ]; then
    INPUT_FILELIST=${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/tmp/das_cache/QCD_PU.txt
    NAME=QCDPU
elif [ "$1" == "ZEEPU" ]; then
    INPUT_FILELIST=${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/tmp/das_cache/ZEE_PU.txt
    NAME=ZEEPU
elif [ "$1" == "ZMMPU" ]; then
    INPUT_FILELIST=${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/tmp/das_cache/ZMM_PU.txt
    NAME=ZMMPU
elif [ "$1" == "TenTauPU" ]; then
    INPUT_FILELIST=${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/tmp/das_cache/TenTau_PU.txt
    NAME=TenTauPU
elif [ "$1" == "NuGunPU" ]; then
    INPUT_FILELIST=${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/tmp/das_cache/NuGun_PU.txt
    NAME=NuGunPU
elif [ "$1" == "conf" ]; then  # special switch for creating conf file, 
    INPUT_FILELIST=${CMSSW_BASE}/src/Validation/RecoParticleFlow/test/tmp/das_cache/NuGun_PU.txt # dummy
    NAME=conf
else
    echo "Argument 1 must be [QCD|QCDPU|ZEEPU|ZMMPU|TenTauPU|NuGunPU|conf] but was $1"
    exit 1
fi

##Which step to do
if [ "$2" == "reco" ]; then
    STEP="RECO"
elif [ "$2" == "dqm" ]; then
    STEP="DQM"
else
    echo "Argument 2 must be [reco|dqm] but was $2"
    exit 1
fi

#skip njob*perjob events
SKIPEVENTS=$(($NJOB * $PERJOB))

#Just print out environment last time for debugging
echo $INPUT_FILELIST $NAME $STEP $SKIPEVENTS
#env

if [ $STEP == "RECO" ]; then

    if [ $NAME == "conf" ]; then
	mkdir -p $NAME
	cd $NAME

	FILENAME=`sed -n "${NJOB}p" $INPUT_FILELIST`
	echo "FILENAME="$FILENAME

	cmsDriver.py step3 --conditions $CONDITIONS -s RAW2DIGI,L1Reco,RECO,RECOSIM,EI,PAT --datatier MINIAODSIM --nThreads $NTHREADS -n -1 --era $ERA --eventcontent MINIAODSIM --geometry=$GEOM --filein step2.root --fileout file:step3_inMINIAODSIM.root --no_exec --python_filename=step3.py $CUSTOM
	
    else
	
	#Start of workflow
	echo "Making subdirectory $NAME"

	if [ -e $NAME ]; then
            echo "directory $NAME exists, aborting"
            exit 1
	fi

	mkdir $NAME
	cd $NAME

	FILENAME=`sed -n "${NJOB}p" $INPUT_FILELIST`
	echo "FILENAME="$FILENAME
	#Run the actual CMS reco with particle flow.
	echo "Running step RECO" 
	cmsDriver.py step3 --conditions $CONDITIONS -s RAW2DIGI,L1Reco,RECO,RECOSIM,EI,PAT --datatier MINIAODSIM --nThreads $NTHREADS -n -1 --era $ERA --eventcontent MINIAODSIM --geometry=$GEOM --filein $FILENAME --fileout file:step3_inMINIAODSIM.root $CUSTOM | tee step3.log  2>&1
   
	#NanoAOD
	#On lxplus, this step takes about 1 minute / 1000 events
	#Can be skipped if doing DQM directly from RECO
	#cmsDriver.py step4 --conditions $CONDITIONS -s NANO --datatier NANOAODSIM --nThreads $NTHREADS -n $N --era $ERA --eventcontent NANOAODSIM --filein file:step3_inMINIAODSIM.root --fileout file:step4.root > step4.log 2>&1

    fi
	
elif [ $STEP == "DQM" ]; then
    echo "Running step DQM" 

    cd $NAME
    
    #get all the filenames and make them into a python-compatible list of strings
    #STEP3FNS=`ls -1 step3*MINIAODSIM*.root | sed 's/^/"file:/;s/$/",/' | tr '\n' ' '`
    du step3*MINIAODSIM*.root | grep -v "^0" | awk '{print $2}' | sed 's/^/file:/' > step3_filelist.txt
    cat step3_filelist.txt 

    #Run the DQM sequences (PF DQM only)
    #override the filenames here as cmsDriver does not allow multiple input files and there is no easy way to merge EDM files
    cmsDriver.py step5 --conditions $CONDITIONS -s DQM:@pfDQM --datatier DQMIO --nThreads $NTHREADS --era $ERA --eventcontent DQM --filein filelist:step3_filelist.txt --fileout file:step5.root -n -1 2>&1 | tee step5.log

    #Harvesting converts the histograms stored in TTrees to be stored in folders by run etc
    cmsDriver.py step6 --conditions $CONDITIONS -s HARVESTING:@pfDQM --era $ERA --filetype DQM --filein file:step5.root --fileout file:step6.root 2>&1 | tee step6.log
fi

#echo "Exit code was $?"
#tail -n3 *.log

cd ..

find . -name "*"
