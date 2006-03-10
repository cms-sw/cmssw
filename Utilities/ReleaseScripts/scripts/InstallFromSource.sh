#! /bin/bash

if [ $# -ne 1 ] ; then 
    echo "Usage: InstallFromSource.sh [CMSSW_version]";
    exit 1;
fi



cmssw_version=$1

echo Installing $cmssw_version;

CVSROOT=:kserver:cmscvs.cern.ch:/cvs_server/repositories/CMSSW

cvs co -r $cmssw_version config

if [ $? -ne 0 ]; then
  echo "Could not checkout config for CMSSW version $cmssw_version" ;
  exit 1;
fi

if [ ! -f config/bootsrc ] ; then
  echo "could not find bootsrc";
  exit 1;
fi

scramv1 project -b config/bootsrc

echo "Finished configuring scram area "

export SCRAM_NOSYMCHECK=true

cd $cmssw_version;

scramv1 b -v -k release-build > logs/slc3_ia32_gcc323/release-build.log 2> logs/slc3_ia32_gcc323/release-build-errors.log

#scramv1 b release-freeze;
eval `scramv1 runtime -sh`
SealPluginRefresh >& logs/slc3_ia32_gcc323/SealPluginRefresh.log;

scramv1 b doc >& logs/slc3_ia32_gcc323/docgen.log ;
scramv1 install;

# freeze
cd ..;
find CMSSW_0_5_0 -type d -exec fs setacl {} -acl system:anyuser rl \;
