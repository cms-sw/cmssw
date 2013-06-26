#!/bin/bash

referencerelease="CMSSW_2_2_3"
basedir="/afs/cern.ch/cms/Physics/muon/CMSSW/Performance/RecoMuon/MuonIdentification"

if [ "$1" == "-h" ] || [ "$1" == "--help" ] || [ $# -gt 6 ]; then
   echo "Usage: muonIdVal.sh [dqmFilename1 = dqm file in \$PWD] [dqmFilename2 = dqm file from ${referencerelease}] [make2DPlots = 1] [printPng = 1] [printHtml = 0] [printEps = 0]"
   echo
   echo "       dqmFilename1 could be the file you want to validate, it's plots are drawn first anyway"
   echo "       dqmFilename2 could be the file you want to use as a reference, it's plots are drawn second anyway"
   echo
   exit 1
fi

args=("$@")

# Initialize arguments to pass to muonIdVal.C
if [ "${args[0]}" == "" ]; then
   args[0]=""
fi
if [ "${args[1]}" == "" ]; then
   args[1]=""
fi
if [ "${args[2]}" == "" ]; then
   args[2]="1"
fi
if [ "${args[3]}" == "" ]; then
   args[3]="1"
fi
if [ "${args[4]}" == "" ]; then
   args[4]="0"
fi
if [ "${args[5]}" == "" ]; then
   args[5]="0"
fi

# Make sure we are in a CMSSW release because we need ROOT
cmsswrelease=$(echo ${CMSSW_BASE} | awk '{print substr($0,match($0,/\/[^\/]*$/)+1,length($0))}')
if [ "${cmsswrelease}" == "" ]; then
   echo "Error: unable to determine CMSSW release; please eval \`scramv1 ru -sh\`"
   exit 2
fi

# Make sure we can find $dqmFilename1 if specified, and the dqm file in $PWD otherwise
if [ "${args[0]}" != "" ]; then
   if [ ! -f "${args[0]}" ]; then
      echo "Error: unable to find ${args[0]}"
      exit 3
   fi
else
   filename1=$(find ${PWD} -name "DQM*.root")
   if [ "$filename1" == "" ]; then
      echo "Error: unable to find dqm file in the present working directory, i.e. ${PWD}/DQM*.root"
      exit 4
   fi
   args[0]="${filename1}"
fi
sample=$(echo ${args[0]} | sed 's/^.*Muons__MuonIdVal__\(.*\)\.root$/\1/')

# Make sure we can find $dqmFilename1 if specified, and the reference release dqm file otherwise
if [ "${args[1]}" != "" ]; then
   if [ ! -f "${args[1]}" ]; then
      echo "Error: unable to find ${args[1]}"
      exit 5
   fi
else
   filename2=$(find ${basedir}/${referencerelease}/${sample} -name "DQM*.root")
   if [ "$filename2" == "" ]; then
       echo "Error: unable to find reference release dqm file, i.e. ${basedir}/${referencerelease}/${sample}/DQM*.root"
       exit 6
   fi
   args[1]="${filename2}"
fi

# Make target directory for this CMSSW release
target="${basedir}/${cmsswrelease}/${sample}"
mkdir -p ${target}
if [ $? -ne 0 ]; then
   echo "Error: unable to create $target, or maybe it already exists"
   exit 7
fi
mkdir -p ${target}/TrackerMuons
if [ $? -ne 0 ]; then
   echo "Error: unable to create $target/TrackerMuons, or maybe it already exists"
   exit 8
fi
mkdir -p ${target}/GlobalMuons
if [ $? -ne 0 ]; then
   echo "Error: unable to create $target/GlobalMuons, or maybe it already exists"
   exit 9
fi

echo "Processing ${cmsswrelease} MuonIdVal..."
echo "   File to validate: ${args[0]}"
echo "   Reference file  : ${args[1]}"
echo "   Target directory: $target"

# Run the muonIdVal.C ROOT macro
echo ".L muonIdVal.C++
muonIdVal(\"${args[0]}\", \"${args[1]}\", ${args[2]}, ${args[3]}, ${args[4]}, ${args[5]})" | root -b -l
if [ $? -ne 0 ]; then
   echo "Error: unable to run root?"
   exit 10
fi

# Move pngs and whatnots to target directory
echo "Moving pngs and whatnots to ${target}..."
mv tm_*.png ${target}/TrackerMuons/
if [ $? -ne 0 ]; then
   echo "Error: failed to move tm_*.png to ${target}/TrackerMuons/"
   exit 11
fi
mv gm_*.png ${target}/GlobalMuons/
if [ $? -ne 0 ]; then
   echo "Error: failed to move gm_*.png to ${target}/GlobalMuons/"
   exit 12
fi
# Is there anything left to move?
if [ "`ls *.png`" != "" ]; then
   mv *.png ${target}/
   if [ $? -ne 0 ]; then
      echo "Error: failed to move *.png to ${target}/"
      exit 13
   fi
fi

cp ${args[0]} ${target}/
if [ $? -ne 0 ]; then
   echo "Error: failed to copy ${args[0]} to ${target}/"
   exit 14
fi
