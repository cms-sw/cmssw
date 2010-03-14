#! /bin/csh

set RefRelease=$1

if ($#argv > 0) then

echo Copying and removing reference histograms for release $RefRelease

tar cvzf ${RefRelease}.tgz /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles/${RefRelease}
rm -r /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles/${RefRelease}
rfcp ${RefRelease}.tgz /castor/cern.ch/cms/Validation/TrackerValidation
else
    echo "provide a version in format CMSSW_X_Y_Z"
endif
