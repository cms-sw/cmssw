#!/bin/bash

if [ $# -ne 2 ]; then
    echo "usage: ./compare.sh CMSSW_RELEASE_1 CMSSW_RELEASE_2"
    exit 1
fi
release1=$1
release2=$2
#dqmdir="/tas03/home/jribnik/devel/muonidval/dqm/fastsim"
dqmdir="/tas03/home/jribnik/devel/muonidval/dqm"

echo "Reticulating spline for $release1/$release2 comparison..."
#Format of a standard RECO DQM file
#DQM_V0001_R000000001__RelValSingleMuPt100__CMSSW_3_6_0_pre1-MC_3XY_V21-v2__GEN-SIM-RECO.root
#Format of a FastSim DQM file
#DQM_V0001_R000000001__RelValSingleMuPt100__CMSSW_3_6_0_pre1-MC_3XY_V21_FastSim-v1__GEN-SIM-DIGI-RECO.root

file1s=(`find -L $dqmdir -maxdepth 1 -name "DQM_V0001_R000000001__*__${release1}-*__*.root"`)
if [ ${#file1s[@]} -eq 0 ]; then
    echo "No samples found for $release1. Exiting."
    exit 2
fi

for file1 in ${file1s[@]}; do
    isfastsim=0
    echo $file1 | grep FastSim >/dev/null
    if [ $? -eq 0 ]; then
        isfastsim=1
    fi

    sample=`echo $file1 | awk -F'__' '{print $2}'`
    vfile1=`echo $file1 | awk -F'__' '{print $3}' | awk -F'-' '{print $3}'`
    vfile2=""
    echo "Found $sample for ${release1}-$vfile1..."

    # Means we did not find a comparable
    # sample in the second release
    gosolo=0

    file2s=(`find -L $dqmdir -maxdepth 1 -name "DQM_V0001_R000000001__${sample}__${release2}-*-${vfile1}__*.root"`)
    if [ ${#file2s[@]} -eq 0 ]; then
        # Could be that the given v does not exist
        # so we'll compare to v1
        file2s=(`find $dqmdir -maxdepth 1 -name "DQM_V0001_R000000001__${sample}__${release2}-*-v1__*.root"`)
    fi
    if [ ${#file2s[@]} -ne 1 ]; then
        echo "but not in $release2. Making non-comparison plots."
        gosolo=1
    else
        vfile2=$(echo ${file2s[0]} | awk -F'__' '{print $3}' | awk -F'-' '{print $3}')
        echo "and in ${release2}-${vfile2}. Making comparison plots."
    fi

    newdir="${release1}-${vfile1}/$sample"
    if [ $gosolo -eq 0 ]; then
        newdir="${release1}-${vfile1}_vs_${release2}-${vfile2}/$sample"
    fi
    mkdir -p $newdir
    cd $newdir
    cp ../../muonIdVal.C ./
    cmd="root -b -l -q 'muonIdVal.C++(\"$file1\")'"
    if [ $gosolo -eq 0 ]; then
        cmd="root -b -l -q 'muonIdVal.C++(\"$file1\",\"${file2s[0]}\")'"
    fi
    eval $cmd
    rm muonIdVal.C muonIdVal_C.d muonIdVal_C.so
    mkdir GlobalMuons TrackerMuons GlobalMuonsNotTrackerMuons TrackerMuonsNotGlobalMuons
    mv gm_*.png GlobalMuons
    mv tm_*.png TrackerMuons
    mv gmntm_*.png GlobalMuonsNotTrackerMuons
    mv tmngm_*.png TrackerMuonsNotGlobalMuons
    cd -
done
