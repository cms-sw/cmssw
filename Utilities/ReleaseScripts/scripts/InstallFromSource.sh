#!/bin/bash

# 
# Simple script to install CMSSW from sources in one go
#
# Author $Id: InstallFromSource.sh,v 1.6 2009/02/06 08:05:48 andreasp Exp $


cmssw_release_area=/afs/cern.ch/cms/Releases/CMSSW

if [ $# -ne 2 ] ; then 
    echo "Usage: InstallFromSource.sh --release|--prerelease [CMSSW_version]";
    exit 1;
fi

case $1 in 

    --release ) 
      is_release=1;;

    --prerelease )
      is_release=0;;
    *)
      echo "Error, --release or --prerelease must be specified";
      exit 1;

    break;
esac

cmssw_version=$2

echo Installing $cmssw_version;

CVSROOT=:gserver:cmscvs.cern.ch:/cvs_server/repositories/CMSSW

cvs co -r $cmssw_version config

if [ $? -ne 0 ]; then
  echo "Could not checkout config for CMSSW version $cmssw_version" ;
  exit 1;
fi

if [ ! -f config/bootsrc ] ; then
  echo "could not find bootsrc";
  exit 1;
fi


#boot the scram project
scramv1 project -b config/bootsrc

echo "Finished configuring scram area, starting "

#skip symbol checking
export SCRAM_NOSYMCHECK=true


#build
cd $cmssw_version;

# If we are building a prerelease, add debug symbols
if [ $is_release -eq 0 ]; then 
  echo "<flags CXXFLAGS=\"-g\">" >> config/BuildFile ;
fi

scramv1 b -v -k release-build > logs/slc3_ia32_gcc323/release-build.log 2> logs/slc3_ia32_gcc323/release-build-errors.log

#check plugins

#scramv1 b release-freeze;
eval `scramv1 runtime -sh`
SealPluginRefresh >& logs/slc3_ia32_gcc323/SealPluginRefresh.log;

#build doc and install
scramv1 b doc >& logs/slc3_ia32_gcc323/docgen.log ;
scramv1 install;

#create symbolic links
cd ..;
if [ $is_release -eq 1 ] ; then 
    ln -sf `pwd`/$cmssw_version $cmssw_release_area/latest_release ;
else
    ln -sf `pwd`/$cmssw_version $cmssw_release_area/latest_prerelease;
fi

# freeze .. auch, does not work, must be afs admins ...
#cd ..;
#find CMSSW_0_5_0 -type d -exec fs setacl {} -acl system:anyuser rl \;
